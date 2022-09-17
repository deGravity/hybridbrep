from automate import  LinearBlock, BipartiteResMRConv 
from hybridbrep import implicit_part_to_data, ImplicitDecoder
from hybridbrep_cpp import ImplicitPart
import torch
import meshplot as mp
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import torch_geometric as tg
from tqdm.notebook import tqdm
import torch_scatter
import json
import os
from matplotlib import pyplot as plt
import random
from automate import HetData
from reproducibility import create_subset

def data_filter(x):
    return x['num_faces'] < 200 and x['largest_face'] < 20 and x['has_infs'] == False and x['surf_max'] < 5.0 and x['curve_max'] < 5.0

class BRepDS(torch.utils.data.Dataset):
    def __init__(
        self, 
        splits, 
        data_root, 
        mode='train', 
        validate_pct=5, 
        preshuffle=True,
        label_root = None,
        label_ext = None,
        seed = None,
        size = None):

        split = mode
        if mode=='validate':
            split = 'train'

        with open(splits, 'r') as f:
            ids = json.load(f)[split]
        
        if seed and size:
            ids = create_subset(ids, seed, size)

        if preshuffle and mode in ['train','validate']:
            random.shuffle(ids)

        if mode=='train':
            ids = [x for i,x in enumerate(ids) if i % 100 <= (100 - validate_pct)]
        elif mode=='validate':
            ids = [x for i,x in enumerate(ids) if i % 100 > (100 - validate_pct)]
        
        self.all_paths = [os.path.join(data_root, f'{id}.pt') for id in ids]
        self.all_labels = [os.path.join(label_root, f'{id}.{label_ext}') for id in ids] if label_root and label_ext else None

            
    def __getitem__(self, idx):
        data = torch.load(self.all_paths[idx])        
        del data['num_faces']
        del data['largest_face']
        del data['sdf_max']
        del data['curve_max']
        del data['surf_max']
        if self.all_labels:
            with open(self.all_labels[idx], 'r') as f:
                labels = [int(x.strip()) for x in f.readlines()]
            data.labels = torch.tensor(labels).long()
            data.__node_sets__.add('labels')
        return data

    def __len__(self):
        return len(self.all_paths)


class BRepFaceEncoder(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.v_in = LinearBlock(3, hidden_size)
        self.e_in = LinearBlock(15, hidden_size)
        self.l_in = LinearBlock(10, hidden_size)
        self.f_in = LinearBlock(17, hidden_size)
        self.v_to_e = BipartiteResMRConv(hidden_size)
        self.e_to_l = BipartiteResMRConv(hidden_size)
        self.l_to_f = BipartiteResMRConv(hidden_size)
    def forward(self, data):
        v = self.v_in(data.vertex_positions)
        e = torch.cat([data.edge_curves, data.edge_curve_parameters, data.edge_curve_flipped.reshape((-1,1))], dim=1)
        e = self.e_in(e)
        l = self.l_in(data.loop_types.float())
        f = torch.cat([data.face_surfaces, data.face_surface_parameters, data.face_surface_flipped.reshape((-1,1))], dim=1)
        f = self.f_in(f)
        # TODO - incorporate edge-loop data and vert-edge data
        # Potential TODO - heterogenous input of edges and curves based on function type
        e = self.v_to_e(v, e, data.edge_to_vertex[[1,0]])
        l = self.e_to_l(e, l, data.loop_to_edge[[1,0]])
        f = self.l_to_f(l, f, data.face_to_loop[[1,0]])
        return f

class BRepFaceAutoencoder(pl.LightningModule):
    def __init__(self, code_size, hidden_size, decoder_layers):
        super().__init__()
        self.encoder = BRepFaceEncoder(code_size)
        self.decoder = ImplicitDecoder(code_size+2, 4, hidden_size, decoder_layers, use_tanh=False)
    
    def forward(self, data, uv, uv_idx):
        codes = self.encoder(data)
        indexed_codes = codes[uv_idx]
        uv_codes = torch.cat([uv, indexed_codes], dim=1)
        pred = self.decoder(uv_codes)
        return pred
    
    def encode(self, data):
        with torch.no_grad():
            return self.encoder(data)
    
    def testing_loss(self, data):
        with torch.no_grad():
            codes = self.encoder(data)
            repeated_codes = codes.repeat_interleave(data.surface_coords.shape[1],dim=0)
            uvs = data.surface_coords.reshape((-1,2))
            uv_codes = torch.cat([uvs, repeated_codes],dim=1)
            target = torch.cat([data.surface_samples[:,:,:3],data.surface_samples[:,:,-1].unsqueeze(-1)],dim=-1)
            target[torch.isinf(target)] = -0.01
            pred = self.decoder(uv_codes).reshape_as(target)
            loss = torch.nn.functional.mse_loss(pred, target)
        return loss, len(codes)

    def training_step(self, data, batch_idx):
        codes = self.encoder(data)
        repeated_codes = codes.repeat_interleave(data.surface_coords.shape[1],dim=0)
        uvs = data.surface_coords.reshape((-1,2))
        uv_codes = torch.cat([uvs, repeated_codes],dim=1)
        target = torch.cat([data.surface_samples[:,:,:3],data.surface_samples[:,:,-1].unsqueeze(-1)],dim=-1)
        target[torch.isinf(target)] = -0.01
        pred = self.decoder(uv_codes).reshape_as(target)
        loss = torch.nn.functional.mse_loss(pred, target)
        batch_size = data.surface_coords.shape[0]
        self.log('train_loss', loss, batch_size=batch_size, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, data, batch_idx):
        codes = self.encoder(data)
        repeated_codes = codes.repeat_interleave(data.surface_coords.shape[1],dim=0)
        uvs = data.surface_coords.reshape((-1,2))
        uv_codes = torch.cat([uvs, repeated_codes],dim=1)
        target = torch.cat([data.surface_samples[:,:,:3],data.surface_samples[:,:,-1].unsqueeze(-1)],dim=-1)
        pred = self.decoder(uv_codes).reshape_as(target)
        loss = torch.nn.functional.mse_loss(pred, target)
        batch_size = data.surface_coords.shape[0]
        self.log('val_loss', loss, batch_size=batch_size, on_epoch=True, on_step=True)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def visualize_losses(self, data):
        codes = self.encoder(data)
        repeated_codes = codes.repeat_interleave(data.surface_coords.shape[1],dim=0)
        uvs = data.surface_coords.reshape((-1,2))
        uv_codes = torch.cat([uvs, repeated_codes], dim=1)
        target = data.surface_samples[:,:,-1]
        pred = self.decoder(uv_codes).reshape_as(target)
        residual = pred - target

if __name__ == '__main__':
    ds = BRepDS('D:/fusion360segmentation/simple_train_test.json', 'D:/fusion360segmentation/simple_preprocessed', 'train')
    print(f'Train Set size = {len(ds)}')
    ds_val = BRepDS('D:/fusion360segmentation/simple_train_test.json', 'D:/fusion360segmentation/simple_preprocessed', 'validate')
    print(f'Val Set Size = {len(ds_val)}')
    dl = tg.loader.DataLoader(ds, batch_size=8, shuffle=True, num_workers=8, persistent_workers=True)
    dl_val = tg.loader.DataLoader(ds_val, batch_size=1, shuffle=False, num_workers=8, persistent_workers=True)
    model = BRepFaceAutoencoder(64, 1024,4)
    #sd = torch.load('best-gen-implicit-weights.ckpt')['state_dict']
    #model.load_state_dict(sd)
    callbacks = [
            pl.callbacks.ModelCheckpoint(
                monitor='train_loss', save_top_k=1, filename="{epoch}-{train_loss:.6f}",mode="min",
            ),
            pl.callbacks.ModelCheckpoint(
                monitor='val_loss', save_top_k=1, filename="{epoch}-{val_loss:.6f}",mode="min",
            ),
            pl.callbacks.early_stopping.EarlyStopping(
                    monitor='val_loss', mode='min', patience=8
                )
        ]
    logger = TensorBoardLogger('D:/fusion360segmentation/runs/encdec','nofilter64e3')
    trainer = pl.Trainer(gpus=1, max_epochs=-1, track_grad_norm=2, callbacks=callbacks, logger=logger, gradient_clip_val=0.5)
    trainer.fit(model, dl, val_dataloaders=dl_val)