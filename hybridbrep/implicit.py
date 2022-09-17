from math import ceil, sqrt
import torch
from functools import reduce
from hybridbrep_cpp import ImplicitPart
from automate import HetData
import torch
import torch_scatter
import numpy as np
import os

def preprocess_file(file_root, file_id, ext, save_root, samples=500, ref_samples=5000, normalize=True, override=False):
    file_path = os.path.join(file_root, f'{file_id}.{ext}')
    save_path = os.path.join(save_root, f'{file_id}.pt')
    if override or not os.path.exists(save_path):
        ipart = ImplicitPart(file_path, samples, ref_samples, normalize)
        preprocess_implicit_part(ipart, save_path, samples)

def preprocess_implicit_part(ipart, save_path, samples):
    data = implicit_part_to_data(ipart, samples)

    num_faces = len(data.face_surfaces)
    num_curves = len(data.curve_samples)
    if num_curves > 0:
        loop_sizes = torch_scatter.scatter_add(torch.ones_like(data.loop_to_edge[0]), data.loop_to_edge[0])
        face_sizes = torch_scatter.scatter_add(loop_sizes[data.face_to_loop[1]], data.face_to_loop[0])
        largest_face = face_sizes.max().item()
    else:
        largest_face = 0

    has_nans = torch.stack([torch.isnan(data[k]).any() for k in data.keys]).any().item()
    has_infs = torch.stack([torch.isinf(data[k]).any() for k in data.keys]).any().item()
    surf_max = data.surface_samples[:,:,:-1].abs().max().item() if num_faces > 0 else 0.0
    sdf_max = data.surface_samples[:,:,-1].abs().max().item() if num_faces > 0 else 0.0
    curve_max = data.curve_samples.abs().max().item() if num_curves > 0 else 0.0

    data.num_faces = num_faces
    data.largest_face = largest_face
    data.sdf_max = sdf_max
    data.curve_max = curve_max
    data.surf_max = surf_max
    data.has_nans = has_nans
    data.has_infs = has_infs
    data.valid = ipart.valid

    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(data, save_dir)

def implicit_part_to_data(part, samples=500):
    curve_size= int(ceil(sqrt(samples)))
    data = HetData()
    data.bounding_box = torch.tensor(part.bounding_box).float()
    data.face_surfaces = torch.nn.functional.one_hot(torch.tensor(part.face_surfaces, dtype=int),5).long()
    data.face_surface_parameters = torch.tensor(part.face_surface_parameters).float()
    data.face_surface_flipped = torch.tensor(part.face_surface_flipped).long()
    data.loop_types = torch.nn.functional.one_hot(torch.tensor(part.loop_types, dtype=int),10).long()
    data.loop_length = torch.tensor(part.loop_length).float()
    data.edge_curves = torch.nn.functional.one_hot(torch.tensor(part.edge_curves, dtype=int),3).long()
    data.edge_curve_parameters = torch.tensor(part.edge_curve_parameters).float()
    data.edge_curve_flipped = torch.tensor(part.edge_curve_flipped).long()
    data.edge_length = torch.tensor(part.edge_length).float()
    data.vertex_positions = torch.tensor(part.vertex_positions).float()
    data.face_to_face = torch.tensor(part.face_to_face).long()
    data.face_to_loop = torch.tensor(part.face_to_loop).long()
    data.loop_to_edge = torch.tensor(part.loop_to_edge).long()
    data.edge_to_vertex = torch.tensor(part.edge_to_vertex).long()
    data.loop_to_vertex = torch.tensor(part.loop_to_vertex).long()
    # Unsure how to store ordered_loop_edge/flipped since variable length per loop
    # Could do it with an extra indexing layer, but not important unless we need them
    data.edge_to_vertex_is_start = torch.tensor(part.edge_to_vertex_is_start).long()
    data.loop_to_edge_flipped = torch.tensor(part.loop_to_edge_flipped).long()
    data.surface_bounds = torch.tensor(np.stack(part.surface_bounds)).float() if len(part.surface_bounds) > 0 else torch.empty((0,2,2))
    data.surface_coords = torch.tensor(np.stack(part.surface_coords)).float() if len(part.surface_coords) > 0 else torch.empty((0,samples,2))
    data.surface_samples = torch.tensor(np.stack(part.surface_samples)).float() if len(part.surface_samples) > 0 else torch.empty((0,samples,7))
    data.curve_bounds = torch.tensor(np.stack(part.curve_bounds)).float() if len(part.curve_bounds) > 0 else torch.empty((0, 2)).float()
    data.curve_samples = torch.tensor(np.stack(part.curve_samples)).float() if len(part.curve_samples) > 0 else torch.empty((0,curve_size,6)).float()

    data.__edge_sets__ = {
        'face_to_face':['face_surfaces', 'face_surfaces', 'edge_curves'],
        'face_to_loop':['face_surfaces', 'loop_types'],
        'loop_to_edge':['loop_types', 'edge_curves'],
        'edge_to_vertex':['edge_curves', 'vertex_positions'],
        'loop_to_vertex':['loop_types', 'vertex_positions']
    }
    data.__node_sets__ = {
        'bounding_box', 'face_surfaces', 'face_surface_parameters', 'face_surface_flipped',
        'loop_types', 'loop_length', 'edge_curves', 'edge_curve_parameters',
        'edge_curve_flipped', 'edge_length', 'edge_curves', 'edge_curve_parameters',
        'edge_curve_flipped', 'edge_length', 'vertex_positions', 'edge_to_vertex_is_start',
        'loop_to_edge_flipped', 'surface_bounds', 'surface_coords', 'surface_samples',
        'curve_bounds', 'curve_samples'
    }

    return data

class EuclideanMap(torch.nn.Module):
    def __init__(self,
        input_dim,
        output_dim,
        code_dim,
        hidden_size,
        num_layers,
        nonlin = torch.nn.functional.relu,
        use_tanh = False
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.code_dim = code_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.input_size = input_dim + code_dim

        self.decoder = ImplicitDecoder(
            input_size = self.input_size,
            output_size = self.output_dim,
            hidden_size = self.hidden_size,
            input_spacing = 0,
            nonlin = nonlin,
            use_tanh = use_tanh
        )
    def forward(self, x):
        return self.decoder(x)

class ImplicitDecoder(torch.nn.Module):
    def __init__(self,
        input_size = 2,
        output_size = 4,
        hidden_size = 1024,
        num_layers = 4,
        input_spacing = 0,
        nonlin = torch.nn.functional.relu,
        use_tanh = True
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_spacing = input_spacing
        self.nonlin = nonlin
        self.use_tanh = use_tanh

        for layer in range(0, self.num_layers):
            if layer == 0:
                in_size = self.input_size
            elif self.input_spacing == 0 or (layer % self.input_spacing == 0):
                in_size = self.input_size + self.hidden_size
            else:
                in_size = self.hidden_size
            if layer == self.num_layers - 1:
                out_size = self.output_size
            else:
                out_size = self.hidden_size

            setattr(
                self,
                "layer_" + str(layer),
                torch.nn.utils.weight_norm(
                    torch.nn.Linear(in_size, out_size))
            )
    
    def forward(self, x):
        y = x
        for layer in range(self.num_layers):
            if ((self.input_spacing == 0) or (layer % self.input_spacing == 0)) and (layer != 0):
                y = torch.cat([y, x], dim=1)
            y = getattr(self, "layer_" + str(layer))(y)
            if layer < (self.num_layers - 1):
                y = self.nonlin(y)
            elif self.use_tanh:
                y = torch.tanh(y)
        return y
    

# SDF Functions for Testing
class Rectangle:
    def __init__(self, *dims):
        self.dims = dims
    def sdf(self, p):
        b = 0.5*torch.tensor(self.dims, device=p.device, dtype=p.dtype)
        d = torch.abs(p) - b
        outside = torch.linalg.norm(torch.maximum(d, torch.zeros_like(d)), dim=1)
        inside = torch.clamp(torch.max(d, dim=1).values, max=0.)
        return outside + inside
    def __repr__(self):
        return f'Rectangle({self.dims})'

class Circle:
    def __init__(self, r, c):
        self.r = r
        self.c = c
    def sdf(self, p):
        center = torch.tensor(self.c, device=p.device, dtype=p.dtype)
        radius = torch.tensor(self.r, device=p.device, dtype=p.dtype)
        center_dist = torch.linalg.norm(p - center, dim=1)
        return center_dist - radius
    def __repr__(self) -> str:
        return f'Circle({self.r}, {self.c}'

class Union:
    def __init__(self, *shapes):
        self.shapes = shapes
    def sdf(self, p):
        sdfs = map(lambda x: x.sdf(p), self.shapes)
        return reduce(torch.minimum, sdfs)
    def __repr__(self):
        args = ', '.join(map(lambda x: x.__repr__(), self.shapes))
        return f'Union({args})'

class Intersection:
    def __init__(self, *shapes):
        self.shapes = shapes
    def sdf(self, p):
        sdfs = map(lambda x: x.sdf(p), self.shapes)
        return reduce(torch.maximum, sdfs)
    def __repr__(self):
        args = ', '.join(map(lambda x: x.__repr__(), self.shapes))
        return f'Intersection({args})'

class Complement:
    def __init__(self, shape):
        self.shape = shape
    def sdf(self, p):
        return -self.shape.sdf(p)
    def __repr__(self):
        return f'Complement({self.shape.__repr__()})'

class Difference:
    def __init__(self, A, *Bs):
        B = Union(*Bs)
        self.shape = Intersection(A, Complement(B))
    def sdf(self, p):
        return self.shape.sdf(p)
    def __repr__(self):
        return self.shape.__repr__()

class Translate:
    def __init__(self, shape, t):
        self.shape = shape
        self.t = t
    def sdf(self, p):
        T = torch.tensor(self.t, device=p.device, dtype=p.dtype)
        return self.shape.sdf(p - T)
    def __repr__(self):
        return f'Translate({self.shape.__repr__()}, {self.t})'

class Scale:
    def __init__(self, shape, s):
        self.shape = shape
        self.s = s
    def sdf(self, p):
        return self.shape.sdf(p / self.s)
    def __repr__(self):
        return f'Scale({self.shape.__repr__()}, {self.s})'
