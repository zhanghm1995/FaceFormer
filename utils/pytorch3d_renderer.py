'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-02-20 14:30:11
Email: haimingzhang@link.cuhk.edu.cn
Description: The face renderer implemented by Pytorch3d
'''

import os
import os.path as osp
import numpy as np
import cv2
from graphviz import render
import torch
import tempfile
import pytorch3d
import subprocess
from tqdm import tqdm
# Util function for loading meshes
from pytorch3d.io import load_obj
# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    Textures,
    PointLights,
    look_at_view_transform,
    FoVPerspectiveCameras, 
    FoVOrthographicCameras, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesVertex,
    TexturesAtlas,
    PointsRenderer,
    PointsRasterizationSettings,
    PointsRasterizer
)

class FaceRenderer:
    def __init__(self, device, rendered_image_size=800) -> None:
        self.device = device
        self.rendered_image_size = rendered_image_size

        self.face_renderer = self.build_renderer(self.device)
        self._load_template_face("./template/texture_mesh.obj")
    
    def _load_template_face(self, template_face_fname):
        _, faces, _ = load_obj(template_face_fname, device=self.device)
        self.tempalte_face_verts_idx = faces.verts_idx[None] # (1, M, 3)

    def __call__(self, face_vertex):
        verts_rgb = torch.ones_like(face_vertex) # (N, V, 3)
        textures = Textures(verts_rgb=verts_rgb.to(self.device))

        tempalte_face_verts_idx = \
            self.tempalte_face_verts_idx.repeat((face_vertex.shape[0], 1, 1))

        # Initialize the mesh with vertices, faces, and textures.
        # Created Meshes object
        face_mesh = Meshes(
            verts=face_vertex,
            faces=tempalte_face_verts_idx,
            textures=textures)
        
        ## Start rendering
        image = self.face_renderer(face_mesh) * 255.0
        image = image[..., :3].cpu().numpy().astype(np.uint8)
        return image
            
    def build_renderer(self, device):
        # Initialize a camera.
        # With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction. 
        # So we move the camera by 180 in the azimuth direction so it is facing the front of the cow. 
        R, T = look_at_view_transform(-0.6, 0, 180) 
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=30)
        # cameras = FoVPerspectiveCameras(device=device)

        # Define the settings for rasterization and shading. Here we set the output image to be of size
        # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
        # and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that 
        # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
        # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
        # the difference between naive and coarse-to-fine rasterization. 
        raster_settings = RasterizationSettings(
            image_size=self.rendered_image_size, 
            blur_radius=0.0, 
            faces_per_pixel=1, 
        )

        lights = PointLights(device=device, location=[[1.0, 0.9, 1.0]])

        # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will 
        # interpolate the texture uv coordinates for each vertex, sample from a texture image and 
        # apply the Phong lighting model
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=device, 
                cameras=cameras,
                lights=lights
            )
        )
        return renderer

    
