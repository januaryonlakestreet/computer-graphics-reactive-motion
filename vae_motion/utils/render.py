import trimesh
import pyrender
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import imageio
import torch
#from viewer.object import floor, camera, light
def save_mesh_render_gif(sequence,faces,filename="mesh.gif",colour=[0.7,0.7,0.7,1.0]):
    B =sequence.shape[0]
    frames = []
    for frame in range(B):
        frames.append(save_mesh_render(sequence[frame], faces, filename, colour,return_image=True))
    imageio.mimwrite(filename,frames,fps=15)



def save_mesh_render(vertices, faces, filename="mesh.png", color=[0.7, 0.7, 0.7, 1.0],return_image=False):
    """
    Render and save a mesh screenshot.

    vertices: (V,3) numpy or torch tensor
    faces: (F,3) numpy array of face indices
    """
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.detach().cpu().numpy()
    if isinstance(faces, torch.Tensor):
        faces = faces.detach().cpu().numpy()

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual.vertex_colors = color

    scene = pyrender.Scene()
    mesh_node = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    scene.add(mesh_node)

    # Add lights + camera
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=np.eye(4))

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)

    T = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, -4],
        [0, 0, 1, 4],
        [0, 0, 0, 1],
    ])
    R = np.array([
        [1, 0, 0, 0],
        [0, np.cos(np.pi / 4), -np.sin(np.pi / 4), 0],
        [0, np.sin(np.pi / 4), np.cos(np.pi / 4), 0],
        [0, 0, 0, 1],
    ])
    pose = T @ R

    scene.add(camera, pose=pose)
    #floor(scene, "utils/floor.png")


    r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)
    color, _ = r.render(scene)
    r.delete()
    if return_image:
        return color
    else:
        import imageio
        imageio.imwrite(filename, color)
