import os
import platform
if platform.system() != 'Windows':
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
import torch
import torch_geometric as tg
import pytorch_lightning as pl
from torch_scatter import scatter_mean

from automate import PartOptions, Part, PartDataset, SBGCN, LinearBlock, part_to_graph, ArgparseInitialized

import os
import numpy as np
import pyrender
import trimesh


class SimplePartDataModule(pl.LightningDataModule, ArgparseInitialized):
        
    def __init__(
        self,
        data_path : str, 
        normalize : bool = True,
        batch_size : int = 32,
        shuffle : bool = True,
        num_workers : int = 10,
        persistent_workers : bool = True
    ):
        super().__init__()
        self.path = data_path
        self.normalize = normalize
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers and (self.num_workers > 0)


    def setup(self, **kwargs):
        self.train = SimplePartDataset(self.path, self.normalize, 'train')
        self.test = SimplePartDataset(self.path, self.normalize, 'test')
        self.validate = SimplePartDataset(self.path, self.normalize, 'validate')

    def train_dataloader(self):
        return tg.loader.DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers,shuffle=self.shuffle, persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return tg.loader.DataLoader(self.validate, batch_size=self.batch_size, num_workers=self.num_workers,shuffle=False, persistent_workers=self.persistent_workers)

    def test_dataloader(self):
        return tg.loader.DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers,shuffle=False, persistent_workers=self.persistent_workers)

class SimplePartDataset(torch.utils.data.Dataset):
    def __init__(self, path, normalize = True, mode='train'):
        super().__init__()
        self.path = os.path.join(path,mode)
        self.normalize = normalize
        self.mode = mode
        self.files = [os.path.join(self.path,x) for x in os.listdir(self.path) if x.endswith('.pt')]
            
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])

        # Normalization
        if self.normalize:
            face_points = data.face_samples[:,:3,:,:].permute((0,2,3,1)).reshape((-1,3))
            edge_points = data.edge_samples[:,:3,:].permute(0,2,1).reshape((-1,3))
            points = torch.cat([face_points, edge_points],dim=0)
            min_point = points.min(dim=0).values
            max_point = points.max(dim=0).values
            center_point = (min_point + max_point) / 2
            scale = ((max_point - min_point) / 2).max()

        face_types = data.faces[:,:5]#[:,:13]
        face_oris = data.faces[:,-1].reshape((-1,1))
        face_params = data.faces[:,13:-1]
        if self.normalize:
            face_params[:,:3] = (face_params[:,:3] - center_point) / scale
            face_params[:,9] = face_params[:,9] / scale
            toruses = (face_types.argmax(dim=1) == 4)
            face_params[toruses,10] = face_params[toruses,10] / scale # only torus scales last param
        data.faces = torch.cat([face_types, face_params, face_oris],dim=1)

        edge_types =  data.edges[:,:3]#[:,:11]
        edge_params = data.edges[:,11:-1]
        if self.normalize:
            edge_params[:,:3] = (edge_params[:,:3] - center_point) / scale
            edge_params[:,9:] = (edge_params[:,9:] / scale)
        edge_oris = data.edges[:,-1].reshape((-1,1))
        data.edges = torch.cat([edge_types, edge_params, edge_oris],dim=1)

        face_xyz = data.face_samples[:,:3,:,:].permute(0,2,3,1)
        if self.normalize:
            face_xyz = (face_xyz - center_point) / scale
        face_mask = data.face_samples[:,-1:,:,:].permute(0,2,3,1)
        data.face_samples = torch.cat([face_xyz, face_mask],dim=3)

        edge_xyz = data.edge_samples[:,:3,:].permute(0,2,1)
        if self.normalize:
            edge_xyz = (edge_xyz - center_point) / scale
        data.edge_samples = edge_xyz

        # TODO - forgot to normalize vertices!
        return data

class UVPartDataset(PartDataset):
    def __init__(
        self,
        splits_path = '/projects/grail/benjonesnb/cadlab/data-release/splits.json',
        data_dir = '/fast/benjones/data/',
        mode = 'train',
        cache_dir = None,
        num_samples = 10,
        get_mesh = False,
        cache_only = False
    ):
        super().__init__(splits_path, data_dir, mode, cache_dir, None, None)
       
        self.options = None

        self.num_samples = num_samples
        self.get_mesh = get_mesh
        
        if cache_only:
            self.options=None # Hack for GPUs since this won't pickle - only works if the cache is already built!

        # Turn off all non-parametric function features
        self.features.face.parametric_function = True
        self.features.face.parameter_values = True
        self.features.face.orientation = True

        self.features.face.center_of_gravity = False
        self.features.face.bounding_box = False
        self.features.face.center_of_gravity = False
        self.features.face.circumference = False
        self.features.face.exclude_origin = False
        self.features.face.na_bounding_box = False
        self.features.face.surface_area = False
        self.features.face.moment_of_inertia = False

        self.features.edge.bounding_box = False
        self.features.edge.center_of_gravity = False
        self.features.edge.end = False
        self.features.edge.exclude_origin = False
        self.features.edge.orientation = True
        self.features.edge.length = False
        self.features.edge.mid_point = False
        self.features.edge.moment_of_inertia = False
        self.features.edge.na_bounding_box = False
        self.features.edge.start = False
        self.features.edge.t_range = False

        self.features.loop.center_of_gravity = False
        self.features.loop.length = False
        self.features.loop.moment_of_inertia = False
        self.features.loop.na_bounding_box = False

        # Turn off part-level features
        self.features.mcfs = False
        self.features.volume = False
        self.features.surface_area = False
        self.features.center_of_gravity = False
        self.features.bounding_box = False
        self.features.moment_of_inertia = False
        
        # Turn on/off mesh features
        self.features.mesh = self.get_mesh
        self.features.mesh_to_topology = self.get_mesh
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(len(self.part_paths))[idx]]
        if not self.cache_dir is None:
            cache_file = os.path.join(self.cache_dir, self.mode, f'{idx}.pt')
            if os.path.exists(cache_file):
                return torch.load(cache_file)
        part_path = os.path.join(self.data_dir, self.part_paths[idx])
        if part_path.endswith('.pt'):
            part = torch.load(part_path)
        else:
            options = PartOptions()
            options.num_uv_samples = self.num_samples
            options.normalize = False
            options.tesselate = self.get_mesh
            options.collect_inferences = False
            options.default_mcfs = False
            part = Part(part_path, options)

        graph = part_to_graph(part, self.features)
        if not self.cache_dir is None:
            torch.save(graph, cache_file)
        return graph

class UVFitter(pl.LightningModule):
    def __init__(
        self,
        sbgcn_size = 64,
        linear_sizes = (512, 512),
        n_samples = 10,
        f_in = 22,
        l_in = 10,
        e_in = 19,
        v_in = 3
    ):
        super().__init__()
        grid_size = 9*n_samples*n_samples
        self.sbgcn = SBGCN(f_in, l_in, e_in, v_in, sbgcn_size, 0)
        self.lin = LinearBlock(sbgcn_size, *linear_sizes, grid_size, last_linear=True)
        self.loss = torch.nn.MSELoss()
    def forward(self, graph):
        _, _, x_f, _, _, _ = self.sbgcn(graph)
        preds = self.lin(x_f)
        return preds
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    def training_step(self, data, batch_idx):
        target = data.face_samples.reshape((data.face_samples.size(0),-1))
        preds = self(data)
        error = self.loss(preds, target)
        self.log('train_loss', error,batch_size=32)
        return error
    def validation_step(self, data, batch_idx):
        target = data.face_samples.reshape((data.face_samples.size(0),-1))
        preds = self(data)
        error = self.loss(preds, target)
        self.log('val_loss', error,batch_size=32)

        #face_classes = data.faces[:,:13].argmax(dim=1).max()
        #face_errors = torch.nn.functional.mse_loss(preds, target, reduction='none')
        #class_errors = scatter_mean(face_errors, face_classes)
        #for i in range(len(class_errors)):
        #    self.log(f'val_loss_class_{i}', class_errors[i])
    def test_step(self, data, batch_idx):
        target = data.face_samples.reshape((data.face_samples.size(0),-1))
        preds = self(data)
        error = self.loss(preds, target)
        self.log('test_loss', error,batch_size=32)


class UVPartDataModule(pl.LightningDataModule):
    def __init__(
        self,
        splits_path = '/projects/grail/benjonesnb/cadlab/data-release/splits.json',
        data_dir = '/fast/benjones/data/',
        cache_dir = None,
        num_samples = 10,
        get_mesh = False,
        cache_only = False,
        batch_size = 32,
        num_workers = 10,
        shuffle=True
    ):
        super().__init__()
        self.splits_path = splits_path
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.get_mesh = get_mesh
        self.cache_only = cache_only

    def setup(self, **kwargs):
        self.train = UVPartDataset(self.splits_path, self.data_dir, 'train', self.cache_dir, self.num_samples, self.get_mesh, self.cache_only)
        self.test = UVPartDataset(self.splits_path, self.data_dir, 'test', self.cache_dir, self.num_samples, self.get_mesh, self.cache_only)
        self.val = UVPartDataset(self.splits_path, self.data_dir, 'validate', self.cache_dir, self.num_samples, self.get_mesh, self.cache_only)

    def train_dataloader(self):
        return tg.data.DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers,shuffle=self.shuffle)

    def val_dataloader(self):
        return tg.data.DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers,shuffle=False)

    def test_dataloader(self):
        return tg.data.DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers,shuffle=False)


class UVPredSBGCN(pl.LightningModule):
    def __init__(self,
        sbgcn_hidden_size = 128,
        render_hidden_sizes = (1024, 1024),
        pred_edges = True,
        pred_faces = True,
        normal_loss = True,
        metric_loss = True,
        arc_loss = True,
        corner_loss = True,
        sbgcn_k = 0,
        render_size = (800,800),
        render_count = 5
    ):
        super().__init__()

        f_in = 17
        l_in = 10
        e_in = 15
        v_in = 3

        self.pred_edges = pred_edges
        self.pred_faces = pred_faces
        self.sbgcn_hidden_size = sbgcn_hidden_size
        self.sbgcn_k = sbgcn_k
        self.sbgcn = SBGCN(f_in, l_in, e_in, v_in, self.sbgcn_hidden_size, self.sbgcn_k)
        self.render_hidden_sizes = render_hidden_sizes

        self.normal_loss = normal_loss and pred_faces
        self.metric_loss = metric_loss and pred_faces
        self.arc_loss = arc_loss and pred_edges
        self.corner_loss = corner_loss and pred_edges

        self.render_size = render_size
        self.renderer = pyrender.OffscreenRenderer(*self.render_size)
        self.render_count = render_count

        if self.pred_edges:
            self.edge_lin = LinearBlock(self.sbgcn_hidden_size, *self.render_hidden_sizes, 30)
        if self.pred_faces:
            self.surf_lin = LinearBlock(self.sbgcn_hidden_size, *self.render_hidden_sizes, 400)
        
        self.save_hyperparameters()

    def __del__(self):
        self.renderer.delete()
    
    def forward(self, x):
        #e = self.vert_to_edge((self.vert_in(x.vertices), self.edge_in(x.edges)), x.edge_to_vertex[[1,0],:])
        #return torch.tanh(self.edge_conv(e))
        x_t, x_p, x_f, x_l, x_e, x_v = self.sbgcn(x)
        if self.pred_faces:
            f_pred = torch.tanh(self.surf_lin(x_f))
        if self.pred_edges:
            e_pred = torch.tanh(self.edge_lin(x_e))
        if self.pred_faces and self.pred_edges:
            return e_pred, f_pred
        elif self.pred_edges:
            return e_pred
        else:
            return self.pred_faces
    
    def common_step(self, batch, batch_idx, step='train'):
        pred = self(batch)
        if self.pred_edges and self.pred_faces:
            pred_edge, pred_surf = pred
        elif self.pred_edges:
            pred_edge = pred
        else:
            pred_surf = pred

        target_edge = batch.edge_samples
        target_surf = batch.face_samples

        n_samples = batch.edge_samples.shape[1]

        if self.pred_edges:
            pred_edge = pred_edge.reshape(-1,n_samples,3)
            loss_edge = torch.nn.functional.mse_loss(pred_edge, target_edge)
            self.log(f'{step}_loss_edge', loss_edge)

        if self.pred_faces:
            pred_surf = pred_surf.reshape(-1,n_samples,n_samples,4)
            loss_surf = torch.nn.functional.mse_loss(pred_surf, target_surf)
            self.log(f'{step}_loss_surf', loss_surf)
        
        if self.pred_edges and self.pred_faces:
            loss = loss_surf + loss_edge
        elif self.pred_edges:
            loss = loss_edge
        else:
            loss = loss_surf

        if self.normal_loss:
            target_normals = surface_normals(target_surf)
            pred_normals = surface_normals(pred_surf)
            normal_loss = (1.0 - torch.nn.functional.cosine_similarity(target_normals, pred_normals, dim=-1)).mean()
            self.log(f'{step}_loss_normals', normal_loss)
            loss = loss + normal_loss

        if self.metric_loss:
            target_d_u, target_d_v = surface_metric(target_surf)
            pred_d_u, pred_d_v = surface_metric(pred_surf)
            metric_loss = (torch.nn.functional.mse_loss(pred_d_u, target_d_u) + torch.nn.functional.mse_loss(pred_d_v, target_d_v)) / 2
            self.log(f'{step}_loss_metric', metric_loss)
            loss = loss + metric_loss

        if self.arc_loss:
            target_arc = arc_lengths(target_edge)
            pred_arc = arc_lengths(pred_edge)
            arc_loss = torch.nn.functional.mse_loss(pred_arc, target_arc)
            self.log(f'{step}_loss_arc', arc_loss)
            loss = loss + arc_loss

        if self.corner_loss:
            target_corners = cos_corner_angles(target_edge)
            pred_corners = cos_corner_angles(pred_edge)
            corner_loss = torch.nn.functional.mse_loss(pred_corners, target_corners)
            self.log(f'{step}_loss_corner', corner_loss)
            loss = loss + corner_loss
        
        self.log(f'{step}_loss', loss)
        if self.pred_faces and self.pred_edges:
            return loss, (pred_surf, pred_edge)
        elif self.pred_edges:
            return loss, pred_edge
        else:
            return loss, pred_surf

    
    def training_step(self, batch, batch_idx):
        loss, _ = self.common_step(batch, batch_idx, 'train')
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, pred = self.common_step(batch, batch_idx, 'val')
        if batch_idx == 0:
            tensorboard = self.logger.experiment
            comparison = render_batch(batch, pred, self.render_count, self.renderer)
            tensorboard.add_image('val_comparison', comparison, self.current_epoch)
    
    def test_step(self, batch, batch_idx):
        loss, _ = self.common_step(batch, batch_idx, 'test')
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

# Tensorboard Visualization Helpers

def tb_edge_mesh(edge_samples, color=(1.0, .8, .1), r = 0.001, M=3):
    edge_samples = edge_samples.detach()
    starts = edge_samples[:,:-1,:].reshape(-1,3)
    ends = edge_samples[:,1:,:].reshape(-1,3)
    N = starts.shape[0]
    dirs = ends - starts
    dirs = dirs / torch.linalg.norm(dirs,dim=1).reshape((-1,1))
    x = torch.cross(dirs, torch.rand_like(dirs))
    x = x / torch.linalg.norm(x,dim=1).reshape((-1,1))
    y = torch.cross(dirs, x)
    t = torch.arange(0,2*np.pi, 2*np.pi/M)
    M = len(t)
    sint = torch.sin(t)
    cost = torch.cos(t)
    circs = r * (torch.outer(cost, x.flatten()) + torch.outer(sint, y.flatten())).reshape(M,-1,3)
    start_circs = (starts + circs).reshape(-1,3)
    end_circs = (ends + circs).reshape(-1,3)
    edge_verts = torch.cat([start_circs, end_circs],dim=0)
    bottom_tris = torch.tensor([[m*N+n, ((m+1)%M)*N+n, ((m+1)%M)*N+n+N*M] for n in range(N) for m in range(M)])
    top_tris = torch.tensor([[m*N+n, ((m+1)%M)*N+n+M*N, m*N+n+M*N] for n in range(N) for m in range(M)])
    edge_faces = torch.cat([bottom_tris,top_tris],dim=0)

    edge_verts = edge_verts
    edge_faces = edge_faces
    edge_colors = (torch.tensor(color).tile((edge_verts.shape[0], 1)) * 255).byte()
    return edge_verts, edge_faces, edge_colors

def tb_face_mesh(face_samples, color=(.3,.7,.8), w = 0.001):
    face_samples = face_samples.detach()
    xyz = face_samples.reshape(-1,4)[:,:3]
    mask = face_samples.reshape(-1,4)[:,3]
    K = len(mask)
    cube_v = torch.tensor([
        [-w,-w,-w],
        [-w,-w,w],
        [-w,w,-w],
        [-w,w,w],
        [w,-w,-w],
        [w,-w,w],
        [w,w,-w],
        [w,w,w]
    ])
    cube_e = torch.tensor([
        [0,2,1],[1,2,3],
        [0,1,4],[1,5,4],
        [1,3,5],[3,7,5],
        [3,2,6],[3,6,7],
        [2,0,4],[2,4,6],
        [4,5,6],[5,7,6]
    ])
    cubes_v = (xyz.reshape(-1,1,3) + cube_v.reshape(1,8,3)).reshape(-1,3)
    cubes_f = torch.cat([cube_e + 8*n for n in range(K)],dim=0)
    cubes_m = mask.repeat_interleave(8).reshape(-1,1)
    base_color = torch.tensor(color).tile((cubes_v.shape[0], 1))
    white = torch.ones_like(base_color)
    cubes_c = cubes_m * base_color + (1-cubes_m) * white

    cubes_v = cubes_v
    cubes_f = cubes_f
    cubes_c = (cubes_c * 255).byte()

    return cubes_v, cubes_f, cubes_c

def tb_mesh(face_samples, edge_samples, color=(1.0, .8, .1), w=0.001, r=0.001, M=3):
    e_v, e_f, e_c = tb_edge_mesh(edge_samples, color, r, M)
    f_v, f_f, f_c = tb_face_mesh(face_samples, color, w)
    v = torch.cat([e_v, f_v], dim=0)
    f = torch.cat([e_f,f_f + e_v.shape[0]], dim=0)
    c = torch.cat([e_c,f_c],dim=0)
    return v, f, c
    
def tb_comp(face_samples, edge_samples, face_pred, edge_pred, gt_color=(.3,.7,.8), pred_color=(1.0, .8, .1), w=0.001, r=0.001, M=3):
    v_gt, f_gt, c_gt = tb_mesh(face_samples, edge_samples, gt_color)
    v_p, f_p, c_p = tb_mesh(face_pred, edge_pred, pred_color)
    v = torch.cat([v_gt, v_p], dim=0)
    f = torch.cat([f_gt, f_p + v_gt.shape[0]], dim=0)
    c = torch.cat([c_gt, c_p],dim=0)
    return v, f, c

def surface_normals(face_samples):
    r = face_samples[:,2:,1:-1,:3] - face_samples[:,1:-1,1:-1,:3]
    l = face_samples[:,:-2,1:-1,:3] - face_samples[:,1:-1,1:-1,:3]
    d = face_samples[:,1:-1,:-2,:3] - face_samples[:,1:-1,1:-1,:3]
    u = face_samples[:,1:-1,2:,:3] - face_samples[:,1:-1,1:-1,:3]

    n1 = torch.cross(d,r)
    n1m = torch.linalg.norm(n1,dim=-1)
    n1m[n1m == 0] = 1
    n1 = n1 / n1m.unsqueeze(-1)
    n2 = torch.cross(r,u)
    n2m = torch.linalg.norm(n2,dim=-1)
    n2m[n2m == 0] = 1
    n2 = n2 / n2m.unsqueeze(-1)
    n3 = torch.cross(u,l)
    n3m = torch.linalg.norm(n3,dim=-1)
    n3m[n3m == 0] = 1
    n3 = n3 / n3m.unsqueeze(-1)
    n4 = torch.cross(l,d)
    n4m = torch.linalg.norm(n4,dim=-1)
    n4m[n4m == 0] = 1
    n4 = n4 / n4m.unsqueeze(-1)

    normals = (n1 + n2 + n3 + n4) / 4

    return normals

def surface_metric(face_samples):
    d_u = torch.linalg.norm(face_samples[:, 1:, :, :3] - face_samples[:,:-1,:,:3],dim=-1)
    d_v = torch.linalg.norm(face_samples[:,:, 1:, :3] - face_samples[:,:,:-1,:3],dim=-1)
    return d_u, d_v

def arc_lengths(edge_samples):
    return torch.linalg.norm(edge_samples[:,1:,:] - edge_samples[:,:-1,:],dim=-1)

def cos_corner_angles(edge_samples):
    forward = edge_samples[:,2:,:] - edge_samples[:,1:-1,:]
    backward = edge_samples[:,:-2,:] - edge_samples[:,1:-1,:]
    return torch.nn.functional.cosine_similarity(forward, backward, dim=-1)


def render(face_samples, edge_samples, orientations, renderer, backfaces=False):
    quad_mesh = face_mesh(face_samples, orientations)
    _, _, quad_mat = pyrender.Mesh._get_trimesh_props(quad_mesh, smooth=False, material=None)
    mesh = pyrender.Mesh.from_trimesh(quad_mesh,smooth=False)

    ev, ef, ec = tb_edge_mesh(edge_samples, r = 0.005)
    edge_mesh = trimesh.Trimesh(vertices = ev, faces=ef, vertex_colors=ec)
    mesh_edge = pyrender.Mesh.from_trimesh(edge_mesh)

    scene = pyrender.Scene()
    scene.add(mesh)
    scene.add(mesh_edge)
    camera = pyrender.OrthographicCamera(xmag=1.0, ymag=1.0)
    s = np.sqrt(2)/2
    d = np.sqrt(3)/3

    x = torch.tensor([-s, 0, -s])
    y = torch.tensor([s, 0, -s])
    z = torch.tensor([d, -d, d])
    p = torch.tensor([1,-1, 1])
    x = torch.cross(y,z)
    camera_pose = torch.zeros([4,4])
    camera_pose[:3,0] = x
    camera_pose[:3,1] = y
    camera_pose[:3,2] = z
    camera_pose[:3,3] = p
    camera_pose[3,3] = 1

    camera_pose


    scene.add(camera, pose=camera_pose)
    light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                            innerConeAngle=np.pi/16.0,
                            outerConeAngle=np.pi/6.0)
    scene.add(light, pose=camera_pose)
    flags = pyrender.constants.RenderFlags.OFFSCREEN
    if backfaces:
        flags |= pyrender.constants.RenderFlags.SKIP_CULL_FACES
    color, _ = renderer.render(scene,flags=flags)
    return color


def render_comparison(gt_face, gt_edge, pr_face, pr_edge, face_orientations, renderer, backfaces=False):
    pr_face = pr_face.detach().reshape_as(gt_face)
    pr_edge = pr_edge.detach().reshape_as(gt_edge)
    orientations = face_orientations.detach()

    gt_im = render(gt_face, gt_edge, orientations, renderer, backfaces)
    pr_im = render(pr_face, pr_edge, orientations, renderer, backfaces)

    return np.concatenate([gt_im,pr_im],axis=1)

def face_mesh(face_samples, orientations, color=(1.0, .8, .1, 0.5)):
    M = face_samples.shape[0]
    N = face_samples.shape[1]
    
    vertices = face_samples[:,:,:,:3].reshape(-1,3)
    alphas = face_samples[:,:,:,3].flatten()
    quad_faces = torch.stack([
        m*(N**2) + torch.tensor([
            [i+j*N, (i+1)+j*N, (i+1)+(j+1)*N, i+(j+1)*N] for i in range(N-1) for j in range(N-1)
        ])
        for m in range(M)
    ])
    quad_faces[orientations == 1,:,:] = torch.flip(quad_faces[orientations == 1,:,:],dims=[2])
    
    quad_faces = quad_faces.reshape(-1,4)
    quad_face_alphas = alphas[quad_faces].sum(dim=1) / 4
    quad_face_color = torch.tensor(color).tile((quad_faces.shape[0],1))
    quad_face_color[:,3] *= quad_face_alphas
    quad_face_color = (255*quad_face_color).byte()

    quad_face_color = torch.cat([quad_face_color, quad_face_color]) # Must duplicated for bugfix (https://github.com/mikedh/trimesh/issues/1471)

    return trimesh.Trimesh(vertices=vertices, faces=quad_faces, face_colors=quad_face_color)

def render_batch(batch, pred, count, renderer):
    face_ids = batch.batch[batch.face_to_flat_topos[1]].to('cpu')
    edge_ids = batch.batch[batch.edge_to_flat_topos[1]].to('cpu')
    brep_ids = face_ids.unique().to('cpu')[:min(count, len(face_ids.unique()))]
    batch_pred_s, batch_pred_e = pred
    batch_pred_e = batch_pred_e.detach().to('cpu').reshape_as(batch.edge_samples)
    batch_pred_s = batch_pred_s.detach().to('cpu').reshape_as(batch.face_samples)
    batch_orientations = batch.faces[:,-1].to('cpu')

    comparisons = []

    for id in brep_ids:
        face_samples = batch.face_samples[face_ids == id,:,:,:].to('cpu')
        edge_samples = batch.edge_samples[edge_ids == id,:,:].to('cpu')
        face_preds = batch_pred_s[face_ids == id,:,:,:]
        edge_preds = batch_pred_e[edge_ids == id,:,:]
        orientations = batch_orientations[face_ids == id]
        comp_im = render_comparison(face_samples, edge_samples, face_preds, edge_preds, orientations, renderer)
        comparisons.append(torch.from_numpy(comp_im))
    comparisons = torch.cat(comparisons).permute(2,0,1)
    return comparisons