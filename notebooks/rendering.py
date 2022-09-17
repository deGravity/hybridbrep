import os
import platform
if platform.system() != 'Windows':
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

from collections import namedtuple

import numpy as np
import pyrender
from pyrender.constants import RenderFlags

import trimesh
import scipy

from matplotlib import pyplot as plt
from matplotlib.patches import Patch

import json
from zipfile import ZipFile
from automate import Part, PartOptions
from tqdm import tqdm


CameraParams = namedtuple('CameraParams', ['pose','mag'])

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

# Gets the camera angle and magnification for a part
# Optionall optimizes for most faces seen, or best view
# of a particular face
def get_camera_angle(V, F, F_id, optimize=None, renderer=None, return_renderer=False):
    # Normalize
    center = (np.min(V,axis=0) + np.max(V,axis=0))/2
    V = V - center
    max_coord = np.abs(V).max()
    scale = 1 / max_coord
    V = V * scale

    corners = np.array([
        [-1.1, 1.1, -1.1],
        [1.1, 1.1, -1.1],
        [-1.1, 1.1, 1.1],
        [1.1, 1.1, 1.1]
    ])

    init_renderer = (renderer is None) and ((optimize is not None) or return_renderer) 
    if init_renderer:
        renderer = pyrender.OffscreenRenderer(
            viewport_width=200, 
            viewport_height=200, 
            point_size=1.0)
    
    if optimize is None:
        corners = corners[:1]

    angles = [compute_camera_params(V, F, F_id, c, optimize, renderer) for c in corners]
    angles = sorted(angles, key=lambda x:x[1], reverse=True)

    if init_renderer and not return_renderer:
        renderer.delete()

    if return_renderer:
        return angles[0][0], renderer

    return angles[0][0]

def compute_camera_params(V, F, F_id, pos, optimize=None, renderer=None):
    camera_pose = look_at([0,0,0],pos,[0,1,0])
    V_t = np.linalg.pinv(camera_pose).dot(
        np.concatenate([V, np.ones((len(V),1))],axis=1).T
        )[:3].T
    mag = 1.05 * np.abs(V_t[:,:2]).max()

    metric = 0.
    if optimize is not None:
        assert(renderer is not None)
        face_colors = pallet(F_id)
        mesh = pyrender.Mesh.from_trimesh(
        trimesh.Trimesh(
            vertices=V, 
            faces=F, 
            face_colors=face_colors), 
        smooth=False)

        camera = pyrender.OrthographicCamera(xmag=mag,ymag=mag)
        scene = pyrender.Scene(ambient_light=[1.0,1.0,1.0], bg_color=[1.0,1.0,1.0])
        scene.add(mesh, pose=np.eye(4))
        scene.add(camera, pose=camera_pose)

        image, _ = renderer.render(scene, flags=RenderFlags.FLAT | RenderFlags.SKIP_CULL_FACES)
        pixel_colors = image.reshape((-1,3))

        if isinstance(optimize, int): # maximize view of a particular face
            target_color = pallet(optimize)[0]
            target_visibility = (pixel_colors == target_color).all(axis=1).sum()
            metric = target_visibility
        else: # maximize number of faces seen

            high_pass = (pixel_colors >= 50).all(axis=1)
            low_pass = (pixel_colors < 50+np.array([137,151,173])).all(axis=1)
            band_pass = high_pass & low_pass
            pixel_colors = pixel_colors[band_pass]

            _, counts = np.unique(pixel_colors, axis=0, return_counts=True)
            # ignore spurrious colors
            # heuristic: at least a quarter row worth of pixels
            num_visible = (counts > image.shape[0]/4).sum()
            metric = num_visible
    
    return CameraParams(camera_pose, mag), metric


def pallet(v):
    if isinstance(v, int):
        v = np.array([v])
    v = np.stack([v,v,v],axis=-1)
    v = v*np.array([47,23,97])
    return (v % np.array([137,151,173])) + 50


RendererParams = namedtuple('RendererParams',['width','height'])

def render_segmented_mesh(
    V,F,F_id,
    id_color=None,
    camera_params=None,
    camera_opt=None,
    transparent_bg=True,
    renderer=None,
    render_params=RendererParams(400,400),
    return_renderer=False
):
    
    # Create a renderer if necessary
    created_renderer = False
    if renderer is None:
        renderer = pyrender.OffscreenRenderer(
            viewport_width=render_params.width, 
            viewport_height=render_params.height
        )
        created_renderer = True


    # Normalize Part
    center = (np.min(V,axis=0) + np.max(V,axis=0))/2
    V = V - center
    max_coord = np.abs(V).max()
    scale = 1 / max_coord
    V = V * scale

    # Setup Camera
    if camera_params is None:
        camera_params = get_camera_angle(V, F, F_id, camera_opt, renderer)
    camera = pyrender.OrthographicCamera(xmag=camera_params.mag,ymag=camera_params.mag)

    # Setup Mesh to Render
    
    # Use default colors if no colors or list of color indices given
    if id_color is None:
        id_color = pallet(np.arange(F_id.max()+1))
    elif len(np.shape(id_color)) == 1:
        id_color = pallet(id_color)
    
    face_mesh = pyrender.Mesh.from_trimesh(
        trimesh.Trimesh(
            vertices=V, 
            faces=F, 
            face_colors=id_color[F_id]), 
        smooth=False)


    # Find edges by rendering all faces in different colors then
    # applying a Laplacian filter to find edges

    if face_mesh.is_transparent:
        face_colors = np.ones((len(F_id),4))*145
        face_colors[:,:3] = pallet(F_id)
    else:
        face_colors = pallet(F_id)

    edge_mesh = pyrender.Mesh.from_trimesh(
        trimesh.Trimesh(
            vertices=V, 
            faces=F, 
            face_colors=face_colors), 
        smooth=False)

    scene = pyrender.Scene(ambient_light=[1.0,1.0,1.0], bg_color=[1.0,1.0,1.0])
    scene.add(edge_mesh, pose=np.eye(4))
    scene.add(camera, pose=camera_params.pose)

    edge_image, _ = renderer.render(scene, flags=RenderFlags.FLAT | RenderFlags.SKIP_CULL_FACES)
    laplace_kernel = np.array([[-1.,-1.,-1.],[-1.,8.,-1.],[-1.,-1.,-1.]])
    edge_mask = (
        ( scipy.ndimage.convolve(edge_image[:,:,0],laplace_kernel)
        + scipy.ndimage.convolve(edge_image[:,:,1],laplace_kernel)
        + scipy.ndimage.convolve(edge_image[:,:,2],laplace_kernel)
        ) == 0
    ).astype(int)


    # Rerender with desired face colors
    scene = pyrender.Scene(ambient_light=[1.0,1.0,1.0], bg_color=[1.0,1.0,1.0,0.0])
    scene.add(face_mesh, pose=np.eye(4))
    scene.add(camera, pose=camera_params.pose)

    image, _ = renderer.render(scene, flags=RenderFlags.FLAT | RenderFlags.SKIP_CULL_FACES)# | RenderFlags.RGBA)

    # Superimpose edges onto image
    #guass_vec = np.array([[1,4,6,4,1]])
    #guass_kern = guass_vec.T.dot(guass_vec)/255
    #blurred_edge = scipy.ndimage.convolve((1-edge_mask).astype(np.float64),guass_kern)
    #edge_stack = np.stack([edge_mask]*4,axis=-1)
    #edge_stack[:,:,3] = 255
    image = np.stack([edge_mask]*3,axis=-1)*image

    # Clean up renderer if created
    if created_renderer and not return_renderer:
        renderer.delete()
    
    # Add alpha channel if requested
    if transparent_bg:
        alpha = (1-(image == 255).prod(axis=2))*255
        w,h = alpha.shape
        image = np.concatenate([image, alpha.reshape((w,h,1))],axis=-1)

    if return_renderer:
        return image, renderer

    return image

def get_legend_values(image, color_key):
    channels = color_key.shape[1]
    colors = np.unique(image[:,:,:channels].reshape((-1,channels)),axis=0)
    used_colors = []
    for c in colors:
        for i,ck in enumerate(color_key):
            if all(c==ck):
                used_colors.append(i)
    return sorted(used_colors)


def make_legend(color_key, label_names, ax, **kwargs):
    elements = [
        Patch(facecolor=np.array(color)/255,edgecolor='black',label=name)
        for color, name in zip(color_key, label_names)
    ]
    ax.legend(handles=elements, **kwargs)

def grid_images(images):
    rows,cols,height,width,channels = images.shape
    grid = np.zeros((rows*height,cols*width,channels)).astype(int)
    for r in range(rows):
        for c in range(cols):
            grid[r*height:(r+1)*height,c*width:(c+1)*width,:] = images[r,c,:,:,:]
    return grid

########################################### DEPRECATED #########################################################3

def find_best_angle_from_part(part, cubify=True, normalize=True):
    return find_best_angle(part.mesh.V, part.mesh.F, part.mesh_topology.face_to_topology, cubify, normalize)

def find_best_angle(V, F, FtoT, cubify=True, normalize=True):
    if normalize: # assuming centered part for now
        max_coord = np.abs(V).max()
        scale = 1 / max_coord
        V = V * scale
    bb = np.stack([V.min(axis = 0), V.max(axis=0)])
    face_labels = np.tile(FtoT, (3,1)).T.astype(int)
    mesh = trimesh.Trimesh(vertices=V, faces=F, face_colors=face_labels)
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
    ordering = sorted(enumerate(zip(n_visible_faces, image_areas)), key=lambda x: x[1], reverse=True)
    best_idx = ordering[0][0]
    best_zoom = zooms[best_idx]
    # Rescale to maximize image in frame
    depth = depths[best_idx]
    mask = depth > 0
    mg = np.stack(np.meshgrid(np.linspace(-1,1,800),np.linspace(-1,1,800)),axis=-1)
    zoom_factor = np.abs(mg.reshape((-1,2))[mask.flatten()]).max() * 1.05 # add 5% for some margin
    return candidate_poses[best_idx], zoom_factor * best_zoom

def get_camera_params(index_path, zip_path, split='test'):
    poses = []
    zooms = []
    opts = PartOptions()
    opts.default_mcfs = False
    opts.num_uv_samples = 0
    opts.sample_normals = 0
    opts.sample_tangents = False
    with open(index_path, 'r') as f:
        index = json.load(f)
    parts_list = [index['template'].format(*x) for x in index[split]]
    with ZipFile(zip_path, 'r') as zf:
        for part_path in tqdm(parts_list):
            part = Part(zf.open(part_path).read().decode('utf-8'), opts)
            pose, zoom = find_best_angle_from_part(part, cubify=True)
            poses.append(pose)
            zooms.append(zoom)
    return poses, zooms

def render_part(
        part, camera_pose, zoom, 
        max_labels = None, face_labels=None, cmap='tab20', 
        renderer = None,viewport_width=800, viewport_height=800, point_size=1.0,
        normalize=True
):
    V = part.mesh.V
    if normalize: # assuming centered part for now
        max_coord = np.abs(V).max()
        scale = 1 / max_coord
        V = V * scale
    if not max_labels:
        max_labels = (part.mesh_topology.face_to_topology.max() + 1)
    palette = plt.cm.get_cmap(cmap, lut=max_labels)
    
    tri_labels = part.mesh_topology.face_to_topology
    
    face_colors = (palette(tri_labels)*255).astype(int)
    
    mesh = trimesh.Trimesh(vertices=V, faces=part.mesh.F, face_colors=face_colors)
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
    if face_labels is not None:
        tri_labels = face_labels[tri_labels]
        face_colors = (palette(tri_labels)*255).astype(int)
        mesh = trimesh.Trimesh(vertices=V, faces=part.mesh.F, face_colors=face_colors)
        pmesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0], bg_color=[1.0, 1.0, 1.0])
        scene.add(pmesh, pose=np.eye(4))
        scene.add(cam, pose=camera_pose)
        color, _ = r.render(scene, flags=pyrender.constants.RenderFlags.FLAT | pyrender.constants.RenderFlags.SKIP_CULL_FACES)

    
    color_with_edges = np.stack([edge_mask]*3,axis=-1)*color
    
    if not renderer:
        r.delete()
    
    return color_with_edges

def render_mesh(
        V, F, labels, camera_pose, zoom, 
        max_labels = None, face_labels=None, cmap='tab20', 
        renderer = None,viewport_width=800, viewport_height=800, point_size=1.0
):
    
    if not max_labels:
        max_labels = (labels.max() + 1)
    palette = plt.cm.get_cmap(cmap, lut=max_labels)
    
    tri_labels = labels
    
    face_colors = (palette(tri_labels)*255).astype(int)
    
    mesh = trimesh.Trimesh(vertices=V, faces=F, face_colors=face_colors)
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
        mesh = trimesh.Trimesh(vertices=V, faces=F, face_colors=face_colors)
        pmesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0], bg_color=[1.0, 1.0, 1.0])
        scene.add(pmesh, pose=np.eye(4))
        scene.add(cam, pose=camera_pose)
        color, _ = r.render(scene, flags=pyrender.constants.RenderFlags.FLAT | pyrender.constants.RenderFlags.SKIP_CULL_FACES)

    
    color_with_edges = np.stack([edge_mask]*3,axis=-1)*color
    
    if not renderer:
        r.delete()
    
    return color_with_edges


def render_part2(
        part, camera_pose, zoom, 
        max_labels = None, face_labels=None, cmap='tab20', 
        renderer = None,viewport_width=800, viewport_height=800, point_size=1.0,
        normalize=True
):
    V = part.mesh.V
    if normalize: # assuming centered part for now
        # Center the Part
        center = (np.min(V,axis=0) + np.max(V,axis=0))/2
        V = V - center
        max_coord = np.abs(V).max()
        scale = 1 / max_coord
        V = V * scale
    
    inferred_max_labels = (part.mesh_topology.face_to_topology.max() + 1)
    if max_labels and face_labels is None:
        inferred_max_labels = max_labels
    if not max_labels:
        max_labels = inferred_max_labels

    if isinstance(cmap, str):
        palette = plt.cm.get_cmap(cmap, lut=inferred_max_labels)
    else:
        palette = plt.cm.get_cmap('tab20', lut=inferred_max_labels)
    
    tri_labels = part.mesh_topology.face_to_topology
    
    face_colors = (palette(tri_labels)*255).astype(int)
    
    mesh = trimesh.Trimesh(vertices=V, faces=part.mesh.F, face_colors=face_colors)
    pmesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    
    cam = pyrender.OrthographicCamera(xmag=zoom, ymag=zoom)
    
    scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0], bg_color=[1.0, 1.0, 1.0])
    scene.add(pmesh, pose=np.eye(4))
    scene.add(cam, pose=camera_pose)
    
    if not renderer:
        r = pyrender.OffscreenRenderer(viewport_width=viewport_width, viewport_height=viewport_height, point_size=point_size)
    else:
        r = renderer
    
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
    if face_labels is not None or not isinstance(cmap, str):
        if isinstance(cmap, str):
            palette = plt.cm.get_cmap(cmap, lut=max_labels)
        else:
            palette = cmap
        tri_labels = face_labels[tri_labels]
        face_colors = (np.array(palette(tri_labels))*255).astype(int)
        mesh = trimesh.Trimesh(vertices=V, faces=part.mesh.F, face_colors=face_colors)
        pmesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0], bg_color=[1.0, 1.0, 1.0])
        scene.add(pmesh, pose=np.eye(4))
        scene.add(cam, pose=camera_pose)
        color, _ = r.render(scene, flags=pyrender.constants.RenderFlags.FLAT | pyrender.constants.RenderFlags.SKIP_CULL_FACES)

    
    color_with_edges = np.stack([edge_mask]*3,axis=-1)*color
    
    if not renderer:
        r.delete()
    
    return color_with_edges
def render_mesh2(
        V, F, labels, camera_pose, zoom, 
        max_labels = None, face_ids = None, face_labels=None, cmap='tab20', 
        renderer = None,viewport_width=800, viewport_height=800, point_size=1.0
):
    
    inferred_max_labels = (labels.max() + 1)
    if max_labels and face_labels is None:
        inferred_max_labels = max_labels
    if not max_labels:
        max_labels = inferred_max_labels
    
    if isinstance(cmap, str):
        palette = plt.cm.get_cmap(cmap, lut=inferred_max_labels)
    else:
        palette = plt.cm.get_cmap('tab20', lut=inferred_max_labels)
    
    tri_labels = labels
    
    face_colors = (palette(tri_labels)*255).astype(int)
    
    mesh = trimesh.Trimesh(vertices=V, faces=F, face_colors=face_colors)
    pmesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    
    cam = pyrender.OrthographicCamera(xmag=zoom, ymag=zoom)
    
    scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0], bg_color=[1.0, 1.0, 1.0])
    scene.add(pmesh, pose=np.eye(4))
    scene.add(cam, pose=camera_pose)
    
    if renderer is None:
        r = pyrender.OffscreenRenderer(viewport_width=viewport_width, viewport_height=viewport_height, point_size=point_size)
    else:
        r = renderer
    
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
    if face_labels is not None or not isinstance(cmap, str):
        if isinstance(cmap, str):
            palette = plt.cm.get_cmap(cmap, lut=max_labels)
        else:
            palette = cmap
        tri_labels = face_labels[tri_labels]
        face_colors = (np.array(palette(tri_labels))*255).astype(int)
        mesh = trimesh.Trimesh(vertices=V, faces=F, face_colors=face_colors)
        pmesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0], bg_color=[1.0, 1.0, 1.0])
        scene.add(pmesh, pose=np.eye(4))
        scene.add(cam, pose=camera_pose)
        color, _ = r.render(scene, flags=pyrender.constants.RenderFlags.FLAT | pyrender.constants.RenderFlags.SKIP_CULL_FACES)

    
    color_with_edges = np.stack([edge_mask]*3,axis=-1)*color
    
    if renderer is None:
        r.delete()
    
    return color_with_edges


def render_grid(parts_grid, viewport_width=800, viewport_height=800, point_size=1.0):
    renderer = pyrender.OffscreenRenderer(
        viewport_width=viewport_width, viewport_height=viewport_height, point_size=point_size)
    images = np.stack([
            np.stack([render_part2(*part_spec, 'tab20', renderer) for part_spec in part_row]) 
        for part_row in parts_grid])
    renderer.delete()
    return grid_images(images)