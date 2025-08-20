"""
PyTorch3D renderer for improved SMPL-X mesh visualization
Replaces PyVista renderer with differentiable rendering capabilities
"""

import torch
import numpy as np
import cv2
from typing import Optional, Tuple, List, Dict
import logging

try:
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (
        look_at_view_transform,
        FoVPerspectiveCameras,
        PointLights,
        DirectionalLights,
        Materials,
        RasterizationSettings,
        MeshRenderer,
        MeshRasterizer,
        SoftPhongShader,
        HardPhongShader,
        TexturesVertex,
    )
    from pytorch3d.renderer.blending import BlendParams
    PYTORCH3D_AVAILABLE = True
except ImportError:
    PYTORCH3D_AVAILABLE = False
    logging.warning("PyTorch3D not available. Install pytorch3d package.")


class PyTorch3DRenderer:
    """
    PyTorch3D-based mesh renderer for SMPL-X visualization
    Provides high-quality, differentiable mesh rendering
    """
    
    def __init__(self, 
                 image_size: Tuple[int, int] = (512, 512),
                 device: str = 'cuda',
                 background_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        """
        Initialize PyTorch3D renderer
        
        Args:
            image_size: Output image size (height, width)
            device: Device for rendering
            background_color: RGB background color (0-1 range)
        """
        if not PYTORCH3D_AVAILABLE:
            raise ImportError("PyTorch3D not available. Install pytorch3d package.")
            
        self.device = device
        self.image_size = image_size
        self.background_color = background_color
        
        # Initialize renderer components
        self._setup_renderer()
        
    def _setup_renderer(self):
        """Setup PyTorch3D renderer components"""
        
        # Rasterization settings
        self.raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=None,
            max_faces_per_bin=None,
            perspective_correct=True
        )
        
        # Default camera (can be overridden)
        R, T = look_at_view_transform(
            dist=2.5,
            elev=0,
            azim=0,
            at=((0, 0, 0),),
            up=((0, 1, 0),)
        )
        self.cameras = FoVPerspectiveCameras(
            device=self.device,
            R=R,
            T=T,
            fov=60
        )
        
        # Lighting
        self.lights = PointLights(
            device=self.device,
            location=[[0.0, 1.0, -2.0]],
            ambient_color=[[0.4, 0.4, 0.4]],
            diffuse_color=[[0.6, 0.6, 0.6]],
            specular_color=[[0.1, 0.1, 0.1]]
        )
        
        # Material
        self.materials = Materials(
            device=self.device,
            specular_color=[[0.2, 0.2, 0.2]],
            shininess=10.0
        )
        
        # Shader
        self.shader = SoftPhongShader(
            device=self.device,
            cameras=self.cameras,
            lights=self.lights,
            materials=self.materials,
            blend_params=BlendParams(background_color=self.background_color)
        )
        
        # Complete renderer
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=self.raster_settings
            ),
            shader=self.shader
        )
        
    def render_mesh(self, 
                   vertices: torch.Tensor,
                   faces: torch.Tensor,
                   colors: Optional[torch.Tensor] = None,
                   camera_distance: float = 2.5,
                   camera_elevation: float = 0,
                   camera_azimuth: float = 0) -> np.ndarray:
        """
        Render SMPL-X mesh using PyTorch3D
        
        Args:
            vertices: Mesh vertices [N, 3]
            faces: Mesh faces [F, 3]
            colors: Vertex colors [N, 3] (optional, default: skin color)
            camera_distance: Camera distance from origin
            camera_elevation: Camera elevation angle (degrees)
            camera_azimuth: Camera azimuth angle (degrees)
            
        Returns:
            rendered_image: Rendered image as numpy array [H, W, 3]
        """
        # Convert to tensors and move to device
        if not isinstance(vertices, torch.Tensor):
            vertices = torch.from_numpy(vertices).float()
        if not isinstance(faces, torch.Tensor):
            faces = torch.from_numpy(faces).long()
            
        vertices = vertices.to(self.device)
        faces = faces.to(self.device)
        
        # Ensure batch dimension
        if vertices.dim() == 2:
            vertices = vertices.unsqueeze(0)
        if faces.dim() == 2:
            faces = faces.unsqueeze(0)
            
        # Default skin color if no colors provided
        if colors is None:
            skin_color = torch.tensor([0.8, 0.6, 0.5], device=self.device)  # Skin tone
            colors = skin_color.unsqueeze(0).expand(vertices.shape[0], vertices.shape[1], -1)
        else:
            if not isinstance(colors, torch.Tensor):
                colors = torch.from_numpy(colors).float()
            colors = colors.to(self.device)
            if colors.dim() == 2:
                colors = colors.unsqueeze(0)
                
        # Create mesh with vertex colors
        textures = TexturesVertex(verts_features=colors)
        mesh = Meshes(verts=vertices, faces=faces, textures=textures)
        
        # Setup camera
        R, T = look_at_view_transform(
            dist=camera_distance,
            elev=camera_elevation,
            azim=camera_azimuth,
            device=self.device
        )
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T, fov=60)
        
        # Update renderer with new camera
        self.renderer.rasterizer.cameras = cameras
        self.renderer.shader.cameras = cameras
        
        # Render
        with torch.no_grad():
            rendered_images = self.renderer(mesh)
            
        # Convert to numpy and remove alpha channel
        rendered_image = rendered_images[0, ..., :3].cpu().numpy()
        rendered_image = (rendered_image * 255).astype(np.uint8)
        
        return rendered_image
    
    def render_smplx_mesh(self,
                         smplx_vertices: torch.Tensor,
                         smplx_faces: torch.Tensor,
                         camera_params: Optional[Dict] = None,
                         overlay_image: Optional[np.ndarray] = None,
                         alpha: float = 0.8) -> np.ndarray:
        """
        Render SMPL-X mesh with camera parameters and optional overlay
        
        Args:
            smplx_vertices: SMPL-X mesh vertices [6890, 3] or [10475, 3]
            smplx_faces: SMPL-X mesh faces
            camera_params: Camera parameters (distance, elevation, azimuth)
            overlay_image: Optional background image for overlay
            alpha: Mesh opacity for overlay
            
        Returns:
            rendered_image: Final rendered image
        """
        # Default camera parameters
        if camera_params is None:
            camera_params = {
                'distance': 2.5,
                'elevation': 0,
                'azimuth': 0
            }
            
        # Render mesh
        rendered_mesh = self.render_mesh(
            vertices=smplx_vertices,
            faces=smplx_faces,
            camera_distance=camera_params.get('distance', 2.5),
            camera_elevation=camera_params.get('elevation', 0),
            camera_azimuth=camera_params.get('azimuth', 0)
        )
        
        # Overlay on background image if provided
        if overlay_image is not None:
            # Resize overlay image to match rendered image
            overlay_resized = cv2.resize(overlay_image, self.image_size[::-1])
            
            # Create mask from rendered image (non-black pixels)
            mask = np.any(rendered_mesh > 10, axis=2)
            
            # Blend images
            result = overlay_resized.copy()
            result[mask] = (alpha * rendered_mesh[mask] + 
                          (1 - alpha) * overlay_resized[mask]).astype(np.uint8)
            
            return result
        else:
            return rendered_mesh
    
    def create_mesh_video_frames(self,
                                vertices_sequence: List[torch.Tensor],
                                faces: torch.Tensor,
                                output_dir: str,
                                camera_params: Optional[Dict] = None,
                                overlay_images: Optional[List[np.ndarray]] = None) -> List[str]:
        """
        Create video frames from mesh sequence
        
        Args:
            vertices_sequence: List of vertex tensors for each frame
            faces: Mesh faces (constant across frames)
            output_dir: Directory to save rendered frames
            camera_params: Camera parameters
            overlay_images: Optional background images for each frame
            
        Returns:
            frame_paths: List of saved frame file paths
        """
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        frame_paths = []
        
        for frame_idx, vertices in enumerate(vertices_sequence):
            # Render current frame
            overlay = overlay_images[frame_idx] if overlay_images else None
            rendered_frame = self.render_smplx_mesh(
                smplx_vertices=vertices,
                smplx_faces=faces,
                camera_params=camera_params,
                overlay_image=overlay
            )
            
            # Save frame
            frame_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(frame_path, cv2.cvtColor(rendered_frame, cv2.COLOR_RGB2BGR))
            frame_paths.append(frame_path)
            
        return frame_paths


def create_pytorch3d_renderer(image_size: Tuple[int, int] = (512, 512),
                             device: str = 'cuda') -> Optional[PyTorch3DRenderer]:
    """
    Factory function to create PyTorch3D renderer with error handling
    
    Args:
        image_size: Output image size
        device: Device for rendering
        
    Returns:
        renderer: PyTorch3D renderer instance or None if failed
    """
    if not PYTORCH3D_AVAILABLE:
        logging.warning("PyTorch3D not available - falling back to PyVista renderer")
        return None
        
    try:
        return PyTorch3DRenderer(image_size=image_size, device=device)
    except Exception as e:
        logging.warning(f"Failed to initialize PyTorch3D renderer: {e}. Falling back to PyVista.")
        return None


def compare_renderers(vertices: torch.Tensor,
                     faces: torch.Tensor,
                     pyvista_renderer,
                     pytorch3d_renderer: PyTorch3DRenderer) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compare PyVista and PyTorch3D rendering results
    
    Args:
        vertices: Mesh vertices
        faces: Mesh faces
        pyvista_renderer: PyVista renderer instance
        pytorch3d_renderer: PyTorch3D renderer instance
        
    Returns:
        pyvista_image: Image rendered with PyVista
        pytorch3d_image: Image rendered with PyTorch3D
    """
    # Render with PyTorch3D
    pytorch3d_image = pytorch3d_renderer.render_mesh(vertices, faces)
    
    # Render with PyVista (assuming it has a similar interface)
    try:
        pyvista_image = pyvista_renderer.render_mesh(vertices.cpu().numpy(), faces.cpu().numpy())
    except Exception as e:
        logging.warning(f"PyVista rendering failed: {e}")
        pyvista_image = np.zeros_like(pytorch3d_image)
    
    return pyvista_image, pytorch3d_image