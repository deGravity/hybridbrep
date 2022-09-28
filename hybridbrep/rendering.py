import os
import platform
if platform.system() == 'Linux' and 'microsoft' not in platform.release():
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