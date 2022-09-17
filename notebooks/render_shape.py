import os
import platform
if platform.system() != 'Windows':
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
import pyrender
from matplotlib import pyplot as plt


from train_latent_space import BRepDS, BRepFaceAutoencoder, BRepFaceEncoder
import torch
import numpy as np
import meshplot as mp
import trimesh



def load_model(ckpt_path = 'D:/fusion360segmentation/BRepFaceAutoencoder_64_1024_4.ckpt'):
    model = BRepFaceAutoencoder(64,1024,4)
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['state_dict'])
    return model

def load_test_set(
    index='D:/fusion360segmentation/simple_train_test.json', 
    datadir='D:/fusion360segmentation/simple_preprocessed'
):
    return BRepDS(index, datadir, 'test', preshuffle=False)

def look_at(point, pos, up):
    z = pos - point
    x = np.cross(up, z)
    y = np.cross(z, x)
    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)
    z = z / np.linalg.norm(z)
    pose = np.eye(4)
    pose[:3,0] = x
    pose[:3,1] = y
    pose[:3,2] = z
    pose[:3,3] = pos
    return pose

def find_best_angle(part, cubify=True):
    bb = part.summary.bounding_box
    face_labels = np.tile(part.mesh_topology.face_to_topology, (3,1)).T.astype(int)
    mesh = trimesh.Trimesh(vertices=part.mesh.V, faces=part.mesh.F, face_colors=face_labels)
    pmesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    r = pyrender.OffscreenRenderer(viewport_width=800, viewport_height=800, point_size=1.0)
    scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0], bg_color=[1.0, 1.0, 1.0])
    
    c = bb.sum(axis=0) # bounding box center
    
    if cubify:
        s = np.abs(bb - c).max()
        bb = np.array([
            [c[0]-s,c[1]-s,c[2]-s],
            [c[0]+s,c[1]+s,c[2]+s]
        ])
        
        
    side_positions = [
        [c[0],c[1],bb[0,2]*1.1], # front
        [c[0],c[1],bb[1,2]*1.1], # back
        [bb[0,0]*1.1,c[1],c[2]], # left
        [bb[1,0]*1.1,c[1],c[2]], # right
    ]    
    
    top_corner_positions = [
        [bb[0,0],bb[1,1],bb[0,2]], # front top left
        [bb[1,0],bb[1,1],bb[0,2]], # front top right
        [bb[0,0],bb[1,1],bb[1,2]], # back top left
        [bb[1,0],bb[1,1],bb[1,2]], # back top right
    ]
    
    bottom_corner_positions = [
        [bb[0,0],bb[0,1],bb[0,2]], # front bottom left
        [bb[1,0],bb[0,1],bb[0,2]], # front bottom right   
        [bb[0,0],bb[0,1],bb[1,2]], # back bottom left
        [bb[1,0],bb[0,1],bb[1,2]], # back bottom right
    ]
    
    side_top_position = [
        [c[0],bb[1,1],bb[0,2]*1.1], # front top
        [c[0],bb[1,1],bb[1,2]*1.1], # back top
        [bb[0,0]*1.1,bb[1,1],c[2]], # left top
        [bb[1,0]*1.1,bb[1,1],c[2]], # right top
    ]
    
    side_bottom_position = [
        [c[0],bb[0,1],bb[0,2]*1.1], # front bottom
        [c[0],bb[0,1],bb[1,2]*1.1], # back bottom
        [bb[0,0]*1.1,bb[0,1],c[2]], # left bottom
        [bb[1,0]*1.1,bb[0,1],c[2]], # right bottom
    ]
    
    candidate_positions = top_corner_positions

    if cubify:
        s = np.abs(bb - c).max()
        bb = np.array([
            [c[0]-s,c[1]-s,c[2]-s],
            [c[0]+s,c[1]+s,c[2]+s]
        ])
    
        top_corner_positions = [
            [bb[0,0],bb[1,1],bb[0,2]], # front top left
            [bb[1,0],bb[1,1],bb[0,2]], # front top right
            [bb[0,0],bb[1,1],bb[1,2]], # back top left
            [bb[1,0],bb[1,1],bb[1,2]], # back top right
        ]
        
        candidate_positions += top_corner_positions
    
        
    n_visible_faces = []
    image_areas = []
    candidate_poses = []
    zooms = []
    depths = []
    #candidate_renderings = []
    for pos in candidate_positions:
        camera_pose = look_at(c, pos, [0,1,0])
        candidate_poses.append(camera_pose)
        zoom = 1.2 * np.abs(camera_pose[:3,:3].dot(bb.T)).max()
        zooms.append(zoom)
        cam = pyrender.OrthographicCamera(xmag=zoom, ymag=zoom)
        scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0], bg_color=[1.0, 1.0, 1.0])
        scene.add(pmesh, pose=np.eye(4))
        scene.add(cam, pose=camera_pose)
        color, depth = r.render(scene, flags=pyrender.constants.RenderFlags.FLAT | pyrender.constants.RenderFlags.SKIP_CULL_FACES)
        n_visible_faces.append(len(np.unique(color[:,:,0].flatten()[color[:,:,0].flatten() <= face_labels[:,0].max()])))
        image_areas.append((depth > 0).sum())
        depths.append(depth)
        #candidate_renderings.append(color)
    ordering = sorted(enumerate(zip(v_faces, image_areas)), key=lambda x: x[1], reverse=True)
    best_idx = ordering[0][0]
    best_zoom = zooms[best_idx]
    # Rescale to maximize image in frame
    depth = depths[best_idx]
    mask = depth > 0
    mg = np.stack(np.meshgrid(np.linspace(-1,1,800),np.linspace(-1,1,800)),axis=-1)
    zoom_factor = np.abs(mg.reshape((-1,2))[mask.flatten()]).max() * 1.05 # add 5% for some margin
    return candidate_poses[best_idx], zoom_factor * best_zoom

def render_part(
        part, camera_pose, zoom, 
        max_labels = None, face_labels=None, cmap='tab20', 
        renderer = None,viewport_width=800, viewport_height=800, point_size=1.0
):
    if not max_labels:
        max_labels = (part.mesh_topology.face_to_topology.max() + 1)
    palette = plt.cm.get_cmap(cmap, lut=max_labels)
    
    tri_labels = part.mesh_topology.face_to_topology
    
    face_colors = (palette(tri_labels)*255).astype(int)
    
    mesh = trimesh.Trimesh(vertices=part.mesh.V, faces=part.mesh.F, face_colors=face_colors)
    pmesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    
    cam = pyrender.OrthographicCamera(xmag=zoom, ymag=zoom)
    
    scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0], bg_color=[1.0, 1.0, 1.0])
    scene.add(pmesh, pose=np.eye(4))
    scene.add(cam, pose=camera_pose)
    
    if not renderer:
        r = pyrender.OffscreenRenderer(viewport_width=viewport_width, viewport_height=viewport_height, point_size=point_size)
    
    # Rendering an image with independent face colors. Do this first regardless of overridding
    # color labels in order to compute hard edges for edge rendering
    color, _ = r.render(scene, flags=pyrender.constants.RenderFlags.FLAT | pyrender.constants.RenderFlags.SKIP_CULL_FACES)
    
    # Detect Edges to add to image
    laplace_kernel = np.array([[0, 1, 0],[1, -4, 1],[0, 1, 0]])
    edge_mask = (
        (scipy.ndimage.convolve(color[:,:,0], laplace_kernel) 
         + scipy.ndimage.convolve(color[:,:,1], laplace_kernel) 
         + scipy.ndimage.convolve(color[:,:,2], laplace_kernel)) == 0
    ).astype(int)
    
    # If we have custom face coloring labels, re-render the colors
    if face_labels:
        tri_labels = face_labels[tri_labels]
        face_colors = (palette(tri_labels)*255).astype(int)
        mesh = trimesh.Trimesh(vertices=part.mesh.V, faces=part.mesh.F, face_colors=face_colors)
        pmesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0], bg_color=[1.0, 1.0, 1.0])
        scene.add(pmesh, pose=np.eye(4))
        scene.add(cam, pose=camera_pose)
        color, _ = r.render(scene, flags=pyrender.constants.RenderFlags.FLAT | pyrender.constants.RenderFlags.SKIP_CULL_FACES)

    
    color_with_edges = np.stack([edge_mask]*3,axis=-1)*color
    
    if not renderer:
        r.delete()
    
    return color_with_edges

def predict(data, model, N=100):
    n_faces = data.face_surfaces.shape[0]
    line = torch.linspace(-0.1,1.1,N)
    grid = torch.cartesian_prod(line, line)
    grids = grid.repeat(n_faces,1)
    indices = torch.arange(n_faces).repeat_interleave(N*N, dim=0)
    with torch.no_grad():
        preds = model(data, grids, indices)
    return preds

def plot_part(p):
    e_i = np.concatenate([
        p.mesh_topology.edge_to_topology[:,1],
        p.mesh_topology.edge_to_topology[:,2],
        p.mesh_topology.edge_to_topology[:,0]
    ])
    e_i = e_i[e_i >= 0]
    f_i = p.mesh_topology.face_to_topology
    e = np.concatenate([
        p.mesh.F[:,[0,1]][(p.mesh_topology.edge_to_topology[:,1] >= 0)],
        p.mesh.F[:,[1,2]][(p.mesh_topology.edge_to_topology[:,2] >= 0)],
        p.mesh.F[:,[2,0]][(p.mesh_topology.edge_to_topology[:,0] >= 0)]
    ],axis=0)
    plot = mp.plot(p.mesh.V, p.mesh.F, return_plot=True)
    plot.add_edges(p.mesh.V, e)

def plot_part_data(m, grid_pred, interior = True):
    mask = (grid_pred[:,-1] <= 0)#(grid_pred[:,:,-1] <= 0).flatten()

    positions = grid_pred[:,:3]#.reshape((-1,3))
    dists = grid_pred[:,-1]#.flatten()

    masked_positions = positions[mask] if interior else positions
    masked_dists = dists[mask] if interior else dists

    plot = mp.plot(masked_positions.numpy(), c=masked_dists.numpy(), shading={'point_size':0.1}, return_plot=True)
    plot_edges(m, plot)

def plot_edges(m, plot):
    all_lines = []
    all_points = []
    offset = 0
    for curve in m.curve_samples:
        lines = [(i+offset,i+1+offset) for i in range(len(curve) - 1)]
        lines = np.array(lines)
        all_lines.append(lines)
        all_points.append(curve[:,:3])
        offset += curve.shape[0]
    points = np.concatenate(all_points, 0)
    lines = np.concatenate(all_lines,0)
    plot.add_edges(points, lines)
def plot_part_data(m, grid_pred, interior = True):
    mask = (grid_pred[:,-1] <= 0)#(grid_pred[:,:,-1] <= 0).flatten()

    positions = grid_pred[:,:3]#.reshape((-1,3))
    dists = grid_pred[:,-1]#.flatten()

    masked_positions = positions[mask] if interior else positions
    masked_dists = dists[mask] if interior else dists

    plot = mp.plot(masked_positions.numpy(), c=masked_dists.numpy(), shading={'point_size':0.1}, return_plot=True)
    plot_edges(m, plot)

def preds_to_mesh(preds, N):
    num_faces = int(preds.shape[0] / (N**2))
    v = lambda f,i,j: f*N**2+i*N+j
    tris = np.array([
        [
            [ v(f,i,j),    v(f,i,j+1),    v(f,i+1,j+1) ],
            [ v(f,i,j+1),   v(f,i+1,j+1), v(f,i+1,j)   ],
            [ v(f,i+1,j+1), v(f,i+1,j),   v(f,i,j)     ],
            [ v(f,i+1,j),   v(f,i,j),     v(f,i,j+1)   ]
        ]
        for f in range(num_faces) for i in range(N-1) for j in range(N-1) 
    ]).reshape((-1,3))

    indices = torch.arange(num_faces).repeat_interleave(N*N, dim=0).numpy()

    verts = preds.numpy()[:,:3]
    dists = preds.numpy()[:,-1]

    tri_faces = indices[tris].max(axis=1)
    tri_mask = np.all(dists[tris] <= 0, axis=1)

    return verts, tris[tri_mask], tri_faces[tri_mask]