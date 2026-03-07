import PIL.Image
import numpy as np
import trimesh
import pyrender


class light:
    def __init__(self,scene):
        self.scene = scene
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        scene.add(light, pose=np.eye(4))


class camera:
    def __init__(self,scene):

        self.scene = scene

        # Add camera + light
        # 45° downward tilt about X
        R = np.array([
            [1, 0, 0, 0],
            [0, np.cos(np.pi / 4), -np.sin(np.pi / 4), 0],
            [0, np.sin(np.pi / 4), np.cos(np.pi / 4), 0],
            [0, 0, 0, 1],
        ])

        # put camera at (0, 2, 3) in world space
        T = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, -4],
            [0, 0, 1, 4],
            [0, 0, 0, 1],
        ])

        pose = T @ R

        cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        self.scene.add(cam, pose=pose)


class floor:
    def __init__(self, scene=None, texture_path=None):
        self.scene = scene
        self.texture_path = texture_path
        try:
            image = PIL.Image.open(self.texture_path)
        except:
            image = PIL.Image.open("..\\utils\\floor.png")


        vertices = np.array([
            [-12.5, 0, -12.5],
            [12.5, 0, -12.5],
            [12.5, 0, 12.5],
            [-12.5, 0, 12.5],
        ])
        faces = np.array([
            [0, 2, 1],  # reversed winding
            [0, 3, 2],
        ])

        # UV coordinates (same order as vertices)
        uvs = np.array([
            [0, 0],
            [-1, 0],
            [-1, -1],
            [0, -1],
        ])
        visuals = trimesh.visual.texture.TextureVisuals(uv=uvs * 35, image=image)
        floor_trimesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visuals, process=False)
        R = trimesh.transformations.rotation_matrix(np.deg2rad(90), [1, 0, 0])
        floor_trimesh.apply_transform(R)

        floor_mesh = pyrender.Mesh.from_trimesh(floor_trimesh, smooth=False)
        scene.add(floor_mesh)
