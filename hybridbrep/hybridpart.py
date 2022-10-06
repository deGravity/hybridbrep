import os
import platform
if platform.system() == 'Linux' and 'microsoft' not in platform.release():
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
import pyrender

import torch
import torch_geometric as tg
from pytorch_lightning import LightningModule
from .implicit import ImplicitDecoder
import pytorch_lightning as pl

from torch_geometric.data import Batch
# Candidate GCN Layers that support edge attributes and bipartite graphs
# We would also love to have some form of residual here if possible
from automate import BipartiteResMRConv, LinearBlock
from torch_geometric.nn import (
    GATv2Conv, # Number of attention heads is likely important, lazy
    TransformerConv, # Number of attention heads likely important, lazy, can do weighted residual
    GINEConv, # No residual-like function
    GMMConv, # Guassion-Mixture Model (2016), lazy
    NNConv, # Provide your own NN as input, lazy
    CGConv, # Used for crystal graphs
    GENConv, # From follow-up to DeepGCNs - very general, and lazy
    GeneralConv # From Design Space for GNNs paper, lazy, can do attention w/ multiple heads
)


from automate import HetData
from hybridbrep_cpp import HybridPart
from math import ceil, sqrt
import torch
import numpy as np


from zipfile import ZipFile
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json

class HPart():
    def __init__(self, path, n_samples=500, n_ref_samples=5000, normalize=False, sort_frac=0.5):
        part = HybridPart(path, n_samples, n_ref_samples, normalize, sort_frac)
        data = HetData()
        ### Part Level Stats ###
        data.bounding_box = torch.tensor(part.bounding_box.reshape((1,2,3))).float()
        data.translation = torch.tensor(part.translation).float()
        data.scale = torch.tensor(part.scale).float()
        data.__node_sets__.add('bounding_box')
        data.__node_sets__.add('translation')
        data.__node_sets__.add('scale')
        
        ### Face Encodings ###
        # One-Hot Encode Surface Types -- Non-Simple are all 0s
        face_surfaces = torch.tensor(part.face_surfaces, dtype=int)
        n_surface_types = face_surfaces.max()+1 if len(face_surfaces) > 0 else 1
        n_surface_types = max(5, n_surface_types)
        face_surfaces = torch.nn.functional.one_hot(face_surfaces, n_surface_types).float()
        face_surfaces = face_surfaces[:,:5]
        face_surface_parameters = torch.tensor(part.face_surface_parameters).float()
        face_surface_flipped = torch.tensor(part.face_surface_flipped).reshape((-1,1)).float()
        data.faces = torch.cat([face_surfaces, face_surface_parameters, face_surface_flipped],dim=1).float()
        data.__node_sets__.add('faces')

        ### Edge Encodings ###
        # One-Hot Encode Curve Types -- Nno-Simple are all 0s
        edge_curves = torch.tensor(part.edge_curves, dtype=int)
        n_curve_types = edge_curves.max()+1 if len(edge_curves) > 0 else 1
        n_curve_types = max(3, n_curve_types)
        edge_curves = torch.nn.functional.one_hot(edge_curves, n_curve_types).float()
        edge_curves = edge_curves[:,:3]
        edge_curve_parameters = torch.tensor(part.edge_curve_parameters).float()
        edge_curve_flipped = torch.tensor(part.edge_curve_flipped).reshape((-1,1)).float()
        data.edges = torch.cat([edge_curves, edge_curve_parameters, edge_curve_flipped],dim=1).float()
        data.__node_sets__.add('edges')

        ### Vertex Encodings ###
        data.vertices = torch.tensor(part.vertex_positions).float()
        data.__node_sets__.add('vertices')

        ### Relationships ###
        data.face_to_face = torch.tensor(part.face_to_face).long()
        data.edge_to_face = torch.tensor(part.face_to_edge[[1,0]]).long()
        data.edge_to_face_flipped = torch.tensor(part.face_to_edge_flipped).reshape((-1,1)).float()
        data.vertex_to_edge = torch.tensor(part.edge_to_vertex[[1,0]]).long()
        data.vertex_to_edge_is_start = torch.tensor(part.edge_to_vertex_is_start).reshape((-1,1)).float()
        data.__edge_sets__['face_to_face'] = ['faces', 'faces', 'edges']
        data.__edge_sets__['edge_to_face'] = ['edges', 'faces']
        data.__edge_sets__['vertex_to_edge'] = ['vertices', 'edges']
        data.__node_sets__.add('edge_to_face_flipped')
        data.__node_sets__.add('vertex_to_edge_is_start')


        ### Surface and Curve Samples ###
        curve_size = int(ceil(sqrt(n_samples)))

        data.surface_bounds = torch.tensor(np.stack(part.surface_bounds)).float() if len(part.surface_bounds) > 0 else torch.empty((0,2,2))
        data.surface_coords = torch.tensor(np.stack(part.surface_coords)).float() if len(part.surface_coords) > 0 else torch.empty((0,n_samples,2))
        
        data.surface_samples = torch.tensor(np.stack(part.surface_samples)).float() if len(part.surface_samples) > 0 else torch.empty((0,n_samples,7))
        data.curve_bounds = torch.tensor(np.stack(part.curve_bounds)).float() if len(part.curve_bounds) > 0 else torch.empty((0, 2)).float()
        data.curve_samples = torch.tensor(np.stack(part.curve_samples)).float() if len(part.curve_samples) > 0 else torch.empty((0,curve_size,6)).float()
        # Flip the curve samples to make edge samples
        # Masks and sums flipped and unflipped versions of every edge
        data.curve_samples = (
            data.curve_samples.flip(dims=(1,)).T * part.edge_curve_flipped + 
            data.curve_samples.T * (1-part.edge_curve_flipped)
        ).T.float()

        # Planes, Cylinders, and Lines can have rediculously bad parameterizations
        # (e.g. - origins way out in space) -- canonicalize these by moving
        # them to the center of their respective parameterizations
        
        # This requires moving the origin along a plane or an axis to be
        # in the center of the bounds, and changing the bounds. since
        # UVs are already normalized to 0-1, this does not affect them 
        # (though numerics could still be bad from sampling before)

        # Canonicalize Planes
        planes = data.faces[:,:5].argmax(dim=1) == 0
        plane_bounds = data.surface_bounds[planes]
        plane_params = data.faces[planes,5:-1]
        plane_origins = plane_params[:,:3]
        plane_normals = plane_params[:,3:6]
        plane_ref_dirs = plane_params[:,9:12]
        plane_y_dirs = plane_normals.cross(plane_ref_dirs,dim=1)
        plane_origins_proj = (plane_normals.T*((plane_origins * plane_normals).sum(dim=1))).T
        plane_origin_diffs = plane_origins - plane_origins_proj
        plane_u_shifts = (plane_origin_diffs * plane_ref_dirs).sum(dim=1)
        plane_v_shifts = (plane_origin_diffs * plane_y_dirs).sum(dim=1)
        plane_bounds[:,:,0] = (plane_bounds[:,:,0].T + plane_u_shifts).T
        plane_bounds[:,:,1] = (plane_bounds[:,:,1].T + plane_v_shifts).T
        data.surface_bounds[planes] = plane_bounds
        data.faces[planes,5:8] = plane_origins_proj

        # Canonicalize Cylinders
        cylinders = data.faces[:,:5].argmax(dim=1) == 1
        cylinder_bounds = data.surface_bounds[cylinders]
        cylinder_params = data.faces[cylinders,5:-1]
        cylinder_origins = cylinder_params[:,:3]
        cylinder_axes = cylinder_params[:,6:9]
        cylinder_origins_proj = cylinder_origins - (
            cylinder_axes.T * (cylinder_origins * cylinder_axes).sum(dim=1)).T
        cylinder_v_shifts = ((cylinder_origins - cylinder_origins_proj)*cylinder_axes).sum(dim=1)
        cylinder_bounds[:,:,1] = (cylinder_bounds[:,:,1].T + cylinder_v_shifts).T
        data.surface_bounds[cylinders] = cylinder_bounds
        data.faces[cylinders,5:8] = cylinder_origins_proj

        # Canonicalize Lines
        lines = data.edges[:,:3].argmax(dim=1) == 0
        line_params = data.edges[lines,3:-1]
        line_origins = line_params[:,:3]
        line_dirs = line_params[:,3:6]
        line_origins_proj = line_origins - (
            line_dirs.T * (line_origins * line_dirs).sum(dim=1)).T
        line_bounds = data.curve_bounds[lines]
        line_t_shifts = ((line_origins - line_origins_proj)*line_dirs).sum(dim=1)
        line_bounds = (line_bounds.T + line_t_shifts).T
        data.curve_bounds[lines] = line_bounds
        data.edges[lines,3:6] = line_origins_proj
        
        # TODO - reparameterize rotations
        # https://zhouyisjtu.github.io/project_rotation/rotation.html

        data.__node_sets__.add('surface_bounds')
        data.__node_sets__.add('surface_coords')
        data.__node_sets__.add('surface_samples')
        data.__node_sets__.add('curve_bounds')
        data.__node_sets__.add('curve_samples')

        # Add Mesh Data
        data.V = torch.tensor(part.V).float()
        data.F = torch.tensor(part.F.T).long()
        data.__node_sets__.add('V')
        data.__edge_sets__['F'] = ['V','V','V']

        # Add Loop Node and Topology Information
        loop_types = torch.tensor(part.loop_types, dtype=int)
        loop_types = torch.nn.functional.one_hot(loop_types, 10).float()
        data.loops = loop_types
        data.__node_sets__.add('loops')
        data.edge_to_loop = torch.tensor(part.loop_to_edge[[1,0]]).long()
        data.edge_to_loop_flipped = torch.tensor(part.loop_to_edge_flipped).float()
        data.__node_sets__.add('edge_to_loop_flipped')
        data.__edge_sets__['edge_to_loop'] = ['edges', 'loops']
        data.loop_to_face = torch.tensor(part.face_to_loop[[1,0]]).long()
        data.__edge_sets__['loop_to_face'] = ['loops', 'faces']

        self.data = data
    
    def transform(self, translation, scale):
        pass

    def augment(self):
        pass

    def to_brep(self):
        pass




class HybridPredictor(LightningModule):
    def __init__(self):
        super().__init__()
        self.renderer = None

    def ensure_renderer(self):
        if self.renderer is None:
            self.renderer = pyrender.OffscreenRenderer(
                viewport_width=200,
                viewport_height=200,
                point_size=1.0
            )
        pass

    def destroy_renderer(self):
        if self.renderer is not None:
            self.renderer.delete()

    def forward(self, batch, face_coords, edge_coords=None):
        return self.model(batch, face_coords, edge_coords)

    def mesh_enc_dec(self, batch, n_samples, grid_bounds=(-.1,1.1)):
        face_codes = self.encode_faces(batch)
        edge_codes = self.encode_edges(batch)
        Vs, Fs = self.mesh_decode(face_codes, n_samples, edge_codes, grid_bounds)
        # TODO - label by originating part
        return Vs, Fs

    def mesh_decode(self, face_codes, n_samples=50, edge_codes=None, grid_bounds=(-.1,1.1)):
        decoded = self.grid_decode(face_codes, n_samples, edge_codes, grid_bounds)
        if edge_codes is None:
            face_preds = decoded
        else:
            face_preds, edge_preds = decoded
        Vs = None
        Fs = None
        # TODO - renumber by originating part if reasonable
        return Vs, Fs

    def grid_enc_dec(self, batch, n_samples=50, grid_bounds=(-.1,1.1)):
        face_codes = self.encode_faces(batch)
        edge_codes = self.encode_edges(batch)
        return self.grid_decode(face_codes, n_samples, edge_codes, grid_bounds)
    
    def grid_decode(self, face_codes, n_samples=50, edge_codes=None, grid_bounds=(-.1,1.1)):
        n_faces = face_codes.shape[0]
        face_u = torch.linspace(0, 1, n_samples, device=face_codes.device)
        face_uv = torch.stack([torch.cartesian_prod(face_u, face_u)]*n_faces)
        #face_batch = torch.arange(n_faces).repeat_interleave(3)
        face_preds = self.decode_faces(face_codes, face_uv)
        if edge_codes is not None:
            n_edges = edge_codes.shape[0]
            edge_grid = torch.linspace(0.0,1.0,n_samples,device=edge_codes.device).repeat(n_edges,1)
            edge_preds = self.decode_edges(edge_codes, edge_grid)
            return face_preds, edge_preds
        else:
            return face_preds
        
        

    def eval_common(self, batch, batch_idx):
        # Edge Targets and coordinates
        n_curves, n_c_samples, c_sample_dim = batch.curve_samples.shape
        c_xyz = batch.curve_samples[:,:,:3].reshape((-1,3))
        edge_coords = torch.linspace(0.,1.,n_c_samples,device=batch.faces.device).repeat(n_curves).reshape((n_curves,-1))

        # Face Targets
        n_surfs, n_s_samples, s_sample_dim = batch.surface_samples.shape
        s_xyz = batch.surface_samples[:,:,:3].reshape((-1,3))
        s_m = batch.surface_samples[:,:,s_sample_dim-1].flatten()
        face_coords = batch.surface_coords

        _, face_preds, _, edge_preds = self(batch, face_coords, edge_coords)

        face_preds_xyz = face_preds[:,:3]
        face_preds_m = face_preds[:,3]

        edge_loss = torch.nn.functional.mse_loss(edge_preds, c_xyz)
        face_loss_xyz = torch.nn.functional.mse_loss(face_preds_xyz, s_xyz)
        face_loss_mask = ((face_preds_m - s_m)**2).mean()#torch.nn.functional.mse_loss(face_preds_m, s_m)

        loss = edge_loss + face_loss_xyz + face_loss_mask

        return loss, edge_loss, face_loss_xyz, face_loss_mask

    def training_step(self, batch, batch_idx):
        batch_size = batch.bounding_box.shape[0]
        loss, edge_loss, face_loss_xyz, face_loss_mask = self.eval_common(batch, batch_idx)
        self.log('train_edge_loss', edge_loss, batch_size=batch_size, on_epoch=True, on_step=True)
        self.log('train_face_loss_xyz', face_loss_xyz, batch_size=batch_size, on_epoch=True, on_step=True)
        self.log('train_face_loss_mask', face_loss_mask, batch_size=batch_size, on_epoch=True, on_step=True)
        self.log('train_loss', loss, batch_size=batch_size, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_size = batch.bounding_box.shape[0]
        loss, edge_loss, face_loss_xyz, face_loss_mask = self.eval_common(batch, batch_idx)
        self.log('val_edge_loss', edge_loss, batch_size=batch_size, on_epoch=True, on_step=True)
        self.log('val_face_loss_xyz', face_loss_xyz, batch_size=batch_size, on_epoch=True, on_step=True)
        self.log('val_face_loss_mask', face_loss_mask, batch_size=batch_size, on_epoch=True, on_step=True)
        self.log('val_loss', loss, batch_size=batch_size, on_epoch=True, on_step=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=5e-4)

class GeneralConvEncDec(HybridPredictor):
    def __init__(self, emb_dim=64, dec_dim=1024, dec_layers=4, attn_heads=16):
        super().__init__()
        self.edge_enc = GeneralConv(-1,emb_dim,-1, attention=True, heads=2)
        self.face_enc = GeneralConv(-1,emb_dim,-1, attention=True, heads=attn_heads)
        self.edge_dec = ImplicitDecoder(emb_dim+1, 3, dec_dim, dec_layers, use_tanh=False)
        self.face_dec = ImplicitDecoder(emb_dim+2, 4, dec_dim, dec_layers, use_tanh=False)
    
    def encode_edges(self, batch):
        # Uncomment this and in encode faces to just test decoding overfitting
        #oh= batch.edges#torch.nn.functional.one_hot(torch.arange(len(batch.edges), device=batch.faces.device)).float()
        #oh[:,3:6] = torch.tanh(oh[:,3:6])
        edge_codes = self.edge_enc((batch.vertices,batch.edges),batch.vertex_to_edge,None,batch.vertex_to_edge_is_start)
        #edge_codes = torch.zeros_like(edge_codes)
        #edge_codes[:,:oh.shape[1]] = oh
        return edge_codes

    def encode_faces(self, batch, edge_codes=None):
        if edge_codes is None:
            edge_codes = self.encode_edges(batch)
        #oh = batch.faces#torch.nn.functional.one_hot(torch.arange(len(batch.faces),device=batch.faces.device)).float()
        face_codes = self.face_enc((edge_codes, batch.faces),batch.edge_to_face,None,batch.edge_to_face_flipped)
        #face_codes = torch.zeros_like(face_codes)
        #face_codes[:,:oh.shape[1]] = oh
        return face_codes

    def decode_faces(self, face_codes, face_coords):
        n_surfs, n_s_samples, _ = face_coords.shape
        s_uv = face_coords.reshape((-1,2))
        rep_face_codes = face_codes.repeat_interleave(n_s_samples,dim=0)
        face_dec_input = torch.cat([s_uv, rep_face_codes],dim=1)
        return self.face_dec(face_dec_input)

    def enc_dec(self, batch, face_coords, edge_coords=None):
        edge_codes = self.encode_edges(batch)
        face_codes = self.encode_faces(batch, edge_codes)
        face_preds = self.decode_faces(face_codes, face_coords)
        if edge_coords is None:
            return face_codes, face_preds
        edge_preds = self.decode_edges(edge_codes, edge_coords)
        return face_codes, face_preds, edge_codes, edge_preds

    # edge_coords: [n_edges x n_edge_samples]
    def decode_edges(self, edge_codes, edge_coords):
        n_edges, n_c_samples = edge_coords.shape
        c_t = edge_coords.reshape((-1,1))
        rep_face_codes = edge_codes.repeat_interleave(n_c_samples,dim=0)
        face_dec_input = torch.cat([c_t, rep_face_codes],dim=1)
        return self.edge_dec(face_dec_input)

    def forward(self, batch, face_coords, edge_coords=None):
        return self.enc_dec(batch, face_coords, edge_coords)




class HybridPartDataset(torch.utils.data.Dataset):

    """
    @classmethod
    def preprocess_dataset_parallel(cls, index_path, data_path, tmp_path, preprocessed_path, procs, **args):
        with open(index_path, 'r') as f:
            index = json.load(f)
        keys = [index['template'].format(*key) for s in {'train','test', 'validate'}.intersection(index.keys()) for key in index[s]]
        with ZipFile(data_path, 'r') as zf_data:            
            for key in tqdm(keys, f'Preprocessing {data_path} to {preprocessed_path}'):
                with zf_data.open(key,'r') as f_data:
                    file_data = f_data.read().decode('utf-8')
                data = HPart(file_data, **args).data
                torch.save(data, f_out)
    """

    @classmethod
    def preprocess_dataset(cls, index_path, data_path, preprocessed_path, **args):
        with open(index_path, 'r') as f:
            index = json.load(f)
        keys = [index['template'].format(*key) for s in {'train','test', 'validate'}.intersection(index.keys()) for key in index[s]]
        with ZipFile(data_path, 'r') as zf_data:
            with ZipFile(preprocessed_path, 'w') as zf_preprocessed:
                for key in tqdm(keys, f'Preprocessing {data_path} to {preprocessed_path}'):
                    with zf_data.open(key,'r') as f_data:
                        file_data = f_data.read().decode('utf-8')
                    data = HPart(file_data, **args).data
                    with zf_preprocessed.open(key,'w') as f_out:
                        torch.save(data, f_out)
                    

    def is_ok(zf, key):
        with zf.open(key,'r') as f:
            data = torch.load(f)
        ss_max = data.surface_samples.abs().max() if len(data.surface_samples) > 0 else 0.0
        cs_max = data.surface_samples.abs().max() if len(data.curve_samples) > 0 else 0.0
        return ss_max < torch.inf and cs_max < torch.inf    

    def __init__(self, index_path, preprocessed_path=None, data_path=None, mode='train', val_frac=.05, seed=42, **args):
        
        if not os.path.exists(preprocessed_path):
            HybridPartDataset.preprocess_dataset(index_path, data_path, preprocessed_path, **args)

        with open(index_path, 'r') as f:
            index = json.load(f)

        split = mode
        if mode == 'validate':
            split = 'train'

        keys = [index['template'].format(*key) for key in index[split]]

        if mode in ['train', 'validate']:
            train_keys, val_keys = train_test_split(keys, test_size=val_frac, random_state=seed)
            keys = train_keys if mode=='train' else val_keys
        
        self.preprocessed_data = None
        self.keys = keys
        self.preprocessed_path = preprocessed_path
        self.mode = mode

        # Temporary - bad keys for f360seg preprocessed
        bad_keys = ['s2.0.0/breps/step/23856_6be75f62_0.stp', 's2.0.0/breps/step/57096_6d459f9d_0.stp']
        self.keys = [key for key in keys if key not in bad_keys]

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        if self.preprocessed_data is None:
            self.preprocessed_data = ZipFile(self.preprocessed_path, 'r')
        with self.preprocessed_data.open(self.keys[idx],'r') as f:
            data = torch.load(f)
        return data

from pytorch_lightning.loggers.tensorboard import TensorBoardLogger


def train_hybrid_encoder(
    index_path, 
    preprocessed_path, 
    logdir, 
    logname, 
    embedding_size=64, 
    hidden_size=1024, 
    layers=4, 
    patience=8,
    num_gpus=1,
    batch_size=8
):
    ds = HybridPartDataset(index_path, preprocessed_path, mode='train')
    print(f'Train Set size = {len(ds)}')
    ds_val = HybridPartDataset(index_path, preprocessed_path, mode='validate')
    print(f'Val Set Size = {len(ds_val)}')
    dl = tg.loader.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=8, persistent_workers=True)
    dl_val = tg.loader.DataLoader(ds_val, batch_size=1, shuffle=False, num_workers=8, persistent_workers=True)
    model = GeneralConvEncDec(emb_dim=embedding_size, dec_dim=hidden_size, dec_layers=layers)

    callbacks = [
            pl.callbacks.ModelCheckpoint(
                monitor='train_loss', save_top_k=1, filename="{epoch}-{train_loss:.6f}",mode="min",
            ),
            pl.callbacks.ModelCheckpoint(
                monitor='val_loss', save_top_k=1, filename="{epoch}-{val_loss:.6f}",mode="min",
            ),
            pl.callbacks.early_stopping.EarlyStopping(
                    monitor='val_loss', mode='min', patience=patience
                )
        ]
    logger = TensorBoardLogger(logdir,logname)
    trainer = pl.Trainer(gpus=num_gpus, max_epochs=-1, track_grad_norm=2, callbacks=callbacks, logger=logger, gradient_clip_val=0.5)
    trainer.fit(model, dl, val_dataloaders=dl_val)
