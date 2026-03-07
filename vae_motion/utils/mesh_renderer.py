# ====================================================================
# FIX: Force Matplotlib to use a non-GUI backend (Agg)
# This resolves the 'Unable to register TkChild class' error.
import matplotlib

matplotlib.use('Agg')
# ====================================================================

# NOTE: The pyrender imports below are now protected from GUI conflicts.
import trimesh
import pyrender
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import imageio
import torch
import os  # Necessary for the robust MeshRenderer class structure


# Ensure to use the robust class structure developed in the previous step
# as it solves the core pyrender instability.

class MeshRenderer:
    """
    A robust, reusable renderer for mesh sequences using pyrender.
    This structure resolves the GL context instability.
    """

    def __init__(self, viewport_width=640, viewport_height=480, camera_pose=None, light_intensity=3.0):
        # Initialize OffscreenRenderer once
        try:
            # We explicitly ensure no pyglet/GL context is created repeatedly
            self.r = pyrender.OffscreenRenderer(viewport_width=viewport_width, viewport_height=viewport_height)
        except Exception as e:
            print(f"Error initializing OffscreenRenderer (likely missing OpenGL/driver/headless support): {e}")
            raise

        self.scene = pyrender.Scene()
        self.light_intensity = light_intensity
        self._setup_scene(camera_pose)

    def _setup_scene(self, custom_pose=None):
        """Sets up the static camera and lighting in the scene."""
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=self.light_intensity)
        self.scene.add(light, pose=np.eye(4))

        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)

        # Standard camera pose from your original function
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
        pose = T @ R if custom_pose is None else custom_pose
        self.scene.add(camera, pose=pose)

    def render_frame(self, vertices, faces, color=[0.7, 0.7, 0.7, 1.0]):
        """Renders a single frame and returns the image array."""
        if isinstance(vertices, torch.Tensor):
            vertices = vertices.detach().cpu().numpy()
        if isinstance(faces, torch.Tensor):
            faces = faces.detach().cpu().numpy()

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh_node = pyrender.Mesh.from_trimesh(mesh, smooth=True)
        mesh_node.primitives[0].material.baseColorFactor = color

        node_name = self.scene.add(mesh_node)
        color_image, _ = self.r.render(self.scene)
        self.scene.remove_node(node_name)

        return color_image

    def render_two_characters_frame(self, vertices_a, vertices_b, faces, color_a=[0.7, 0.7, 0.7, 1.0],
                                    color_b=[0.3, 0.3, 0.9, 1.0]):
        if isinstance(vertices_a, torch.Tensor):
            vertices_a = vertices_a.detach().cpu().numpy()
        if isinstance(vertices_b, torch.Tensor):
            vertices_b = vertices_b.detach().cpu().numpy()
        if isinstance(faces, torch.Tensor):
            faces = faces.detach().cpu().numpy()

        mesh_a = trimesh.Trimesh(vertices=vertices_a, faces=faces, process=False)
        mesh_node_a = pyrender.Mesh.from_trimesh(mesh_a, smooth=True)
        mesh_node_a.primitives[0].material.baseColorFactor = color_a

        mesh_b = trimesh.Trimesh(vertices=vertices_b, faces=faces, process=False)
        mesh_node_b = pyrender.Mesh.from_trimesh(mesh_b, smooth=True)
        mesh_node_b.primitives[0].material.baseColorFactor = color_b

        node_a = self.scene.add(mesh_node_a)
        node_b = self.scene.add(mesh_node_b)

        color_image, _ = self.r.render(self.scene)

        self.scene.remove_node(node_a)
        self.scene.remove_node(node_b)

        return color_image

    def save_mesh_twin_render_gif(self, sequence_a,sequence_b, faces, filename="mesh.gif", color=[0.7, 0.7, 0.7, 1.0]):
        """Renders an entire sequence and saves it as a GIF."""
        B = sequence_a.shape[0]
        frames = []
        for frame in range(B):
            frames.append(self.render_two_characters_frame(sequence_a[frame],sequence_b[frame], faces, color))

        imageio.mimwrite(filename, frames, fps=15)

    def save_mesh_render_gif(self, sequence, faces, filename="mesh.gif", color=[0.7, 0.7, 0.7, 1.0]):
        """Renders an entire sequence and saves it as a GIF."""
        B = sequence.shape[0]
        frames = []
        for frame in range(B):
            frames.append(self.render_frame(sequence[frame], faces, color))

        imageio.mimwrite(filename, frames, fps=15)

    def close(self):
        """Cleans up the OffscreenRenderer resources."""
        if hasattr(self, 'r'):
            self.r.delete()

