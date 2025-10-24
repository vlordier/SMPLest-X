#!/usr/bin/env python3
"""
PyVista-based 3D mesh renderer for SMPLest-X.
Replaces matplotlib with proper 3D mesh visualization.
"""

import numpy as np
import pyvista as pv
from pathlib import Path
import cv2
import os
from typing import Tuple, Optional, Union


class PyVistaRenderer:
    """High-quality 3D mesh renderer using PyVista."""
    
    def __init__(self, 
                 window_size: Tuple[int, int] = (800, 800),
                 background_color: str = 'white',
                 lighting: bool = True):
        """
        Initialize PyVista renderer.
        
        Args:
            window_size: (width, height) of render window
            background_color: Background color for renders
            lighting: Whether to use lighting effects
        """
        # Configure PyVista for offscreen rendering
        pv.set_plot_theme('document')
        pv.global_theme.window_size = list(window_size)
        pv.global_theme.background = background_color
        pv.global_theme.lighting = lighting
        pv.global_theme.show_scalar_bar = False
        pv.global_theme.axes.show = False
        pv.global_theme.anti_aliasing = 'fxaa'
        
        # Try to use offscreen rendering
        try:
            pv.start_xvfb()  # For Linux headless
        except:
            pass  # Continue without xvfb (works on Mac/Windows)
        
        self.window_size = window_size
        self.background_color = background_color
        self.lighting = lighting
    
    def render_mesh(self, 
                   vertices: np.ndarray, 
                   faces: np.ndarray,
                   output_path: Optional[str] = None,
                   camera_position: Optional[Union[str, Tuple]] = None,
                   color: str = 'lightblue',
                   show_edges: bool = True,
                   edge_color: str = 'black',
                   smooth_shading: bool = True,
                   opacity: float = 1.0) -> Optional[np.ndarray]:
        """
        Render a 3D mesh using PyVista.
        
        Args:
            vertices: Vertex coordinates (N, 3)
            faces: Face indices (M, 3) 
            output_path: Path to save image (optional)
            camera_position: Camera position ('xy', 'xz', 'yz', 'iso' or custom)
            color: Mesh color
            show_edges: Whether to show mesh edges
            edge_color: Color of mesh edges
            smooth_shading: Use smooth shading
            opacity: Mesh opacity (0-1)
            
        Returns:
            Rendered image as numpy array if output_path is None
        """
        
        # Create PyVista mesh from vertices and faces
        mesh = self._create_pyvista_mesh(vertices, faces)
        
        # Create plotter for offscreen rendering
        plotter = pv.Plotter(off_screen=True, window_size=self.window_size)
        
        # Add mesh to plotter
        plotter.add_mesh(
            mesh, 
            color=color,
            show_edges=show_edges,
            edge_color=edge_color,
            smooth_shading=smooth_shading,
            opacity=opacity,
            lighting=self.lighting
        )
        
        # Set camera position
        if camera_position is not None:
            if isinstance(camera_position, str):
                # Use predefined positions
                if camera_position == 'iso':
                    plotter.view_isometric()
                elif camera_position == 'xy':
                    plotter.view_xy()
                elif camera_position == 'xz': 
                    plotter.view_xz()
                elif camera_position == 'yz':
                    plotter.view_yz()
                elif camera_position == 'front':
                    plotter.camera_position = 'xy'
                elif camera_position == 'side':
                    plotter.camera_position = 'xz'
            else:
                # Custom camera position
                plotter.camera_position = camera_position
        else:
            # Default isometric view
            plotter.view_isometric()
        
        # Fit camera to mesh bounds
        plotter.camera.zoom(1.2)
        
        # Render and capture
        if output_path:
            # Save directly to file
            plotter.screenshot(output_path, transparent_background=False)
            plotter.close()
            return None
        else:
            # Return as numpy array
            image = plotter.screenshot(transparent_background=False, return_img=True)
            plotter.close()
            return image
    
    def render_mesh_wireframe(self,
                            vertices: np.ndarray,
                            faces: np.ndarray, 
                            output_path: Optional[str] = None,
                            camera_position: Optional[Union[str, Tuple]] = None,
                            line_width: int = 1,
                            color: str = 'black') -> Optional[np.ndarray]:
        """
        Render mesh as wireframe.
        
        Args:
            vertices: Vertex coordinates (N, 3)
            faces: Face indices (M, 3)
            output_path: Path to save image (optional)  
            camera_position: Camera position
            line_width: Width of wireframe lines
            color: Wireframe color
            
        Returns:
            Rendered image as numpy array if output_path is None
        """
        
        # Create PyVista mesh
        mesh = self._create_pyvista_mesh(vertices, faces)
        
        # Create plotter
        plotter = pv.Plotter(off_screen=True, window_size=self.window_size)
        
        # Add wireframe mesh
        plotter.add_mesh(
            mesh,
            style='wireframe',
            line_width=line_width,
            color=color,
            lighting=False
        )
        
        # Set camera position
        if camera_position:
            if isinstance(camera_position, str):
                if camera_position == 'iso':
                    plotter.view_isometric()
                elif camera_position == 'xy':
                    plotter.view_xy()
                elif camera_position == 'xz':
                    plotter.view_xz() 
                elif camera_position == 'yz':
                    plotter.view_yz()
            else:
                plotter.camera_position = camera_position
        else:
            plotter.view_isometric()
            
        plotter.camera.zoom(1.2)
        
        # Render
        if output_path:
            plotter.screenshot(output_path, transparent_background=False)
            plotter.close()
            return None
        else:
            image = plotter.screenshot(transparent_background=False, return_img=True)
            plotter.close()
            return image
    
    def render_mesh_with_texture(self,
                               vertices: np.ndarray,
                               faces: np.ndarray,
                               texture_coords: Optional[np.ndarray] = None,
                               texture_image: Optional[str] = None,
                               output_path: Optional[str] = None,
                               camera_position: Optional[Union[str, Tuple]] = None) -> Optional[np.ndarray]:
        """
        Render mesh with texture mapping.
        
        Args:
            vertices: Vertex coordinates (N, 3)
            faces: Face indices (M, 3) 
            texture_coords: UV coordinates (N, 2)
            texture_image: Path to texture image
            output_path: Path to save image (optional)
            camera_position: Camera position
            
        Returns:
            Rendered image as numpy array if output_path is None
        """
        
        # Create PyVista mesh
        mesh = self._create_pyvista_mesh(vertices, faces)
        
        # Add texture coordinates if provided
        if texture_coords is not None:
            # PyVista expects texture coordinates as point data
            mesh.point_data['texture_coords'] = texture_coords
        
        # Create plotter
        plotter = pv.Plotter(off_screen=True, window_size=self.window_size)
        
        # Load texture if provided
        texture = None
        if texture_image and os.path.exists(texture_image):
            texture = pv.read_texture(texture_image)
        
        # Add mesh with texture
        plotter.add_mesh(
            mesh,
            texture=texture,
            smooth_shading=True,
            lighting=self.lighting
        )
        
        # Set camera
        if camera_position:
            if isinstance(camera_position, str) and camera_position == 'iso':
                plotter.view_isometric()
            elif camera_position:
                plotter.camera_position = camera_position
        else:
            plotter.view_isometric()
            
        plotter.camera.zoom(1.2)
        
        # Render
        if output_path:
            plotter.screenshot(output_path, transparent_background=False)
            plotter.close()
            return None
        else:
            image = plotter.screenshot(transparent_background=False, return_img=True)
            plotter.close()
            return image
    
    def _create_pyvista_mesh(self, vertices: np.ndarray, faces: np.ndarray) -> pv.PolyData:
        """
        Create PyVista PolyData from vertices and faces.
        
        Args:
            vertices: Vertex coordinates (N, 3)
            faces: Face indices (M, 3)
            
        Returns:
            PyVista PolyData mesh
        """
        
        # Ensure vertices are float32
        vertices = np.asarray(vertices, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.int32)
        
        # Validate mesh data
        if vertices.shape[1] != 3:
            raise ValueError(f"Vertices must have 3 coordinates, got {vertices.shape[1]}")
        if faces.shape[1] != 3:
            raise ValueError(f"Faces must be triangular, got {faces.shape[1]} vertices per face")
            
        # Check face indices are valid
        if faces.max() >= len(vertices):
            raise ValueError(f"Face indices ({faces.max()}) exceed vertex count ({len(vertices)})")
        if faces.min() < 0:
            raise ValueError(f"Face indices must be non-negative, got {faces.min()}")
        
        # Create faces array in PyVista format: [3, v0, v1, v2, 3, v3, v4, v5, ...]
        pyvista_faces = np.column_stack([
            np.full(len(faces), 3, dtype=np.int32),  # Each face has 3 vertices
            faces
        ]).flatten()
        
        # Create PyVista PolyData
        mesh = pv.PolyData(vertices, pyvista_faces)
        
        return mesh


def render_smplx_mesh(vertices: np.ndarray, 
                     faces: np.ndarray,
                     output_path: str,
                     renderer_type: str = 'shaded',
                     camera_position: str = 'iso',
                     color: str = 'lightblue',
                     window_size: Tuple[int, int] = (800, 800)) -> bool:
    """
    Convenience function to render SMPL-X mesh with PyVista.
    
    Args:
        vertices: Mesh vertices (10475, 3)
        faces: Mesh faces (20908, 3) 
        output_path: Where to save the rendered image
        renderer_type: 'shaded', 'wireframe', or 'both'
        camera_position: Camera view ('iso', 'front', 'side', etc.)
        color: Mesh color
        window_size: Render window size
        
    Returns:
        True if successful
    """
    
    try:
        # Create renderer
        renderer = PyVistaRenderer(window_size=window_size)
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:  # Only create directory if path has a directory component
            os.makedirs(output_dir, exist_ok=True)
        
        if renderer_type == 'wireframe':
            # Render wireframe only
            renderer.render_mesh_wireframe(
                vertices, faces, 
                output_path=output_path,
                camera_position=camera_position,
                line_width=2,
                color='black'
            )
        elif renderer_type == 'both':
            # Render shaded with wireframe overlay
            renderer.render_mesh(
                vertices, faces,
                output_path=output_path, 
                camera_position=camera_position,
                color=color,
                show_edges=True,
                edge_color='black',
                smooth_shading=True,
                opacity=0.8
            )
        else:
            # Default shaded rendering
            renderer.render_mesh(
                vertices, faces,
                output_path=output_path,
                camera_position=camera_position, 
                color=color,
                show_edges=False,
                smooth_shading=True
            )
        
        return True
        
    except Exception as e:
        print(f"❌ PyVista rendering failed: {e}")
        return False


if __name__ == "__main__":
    # Test the renderer with a simple example
    print("🎨 Testing PyVista renderer...")
    
    # Create a simple test mesh (tetrahedron)
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0], 
        [0.5, 0.866, 0.0],
        [0.5, 0.289, 0.816]
    ])
    
    faces = np.array([
        [0, 1, 2],
        [0, 1, 3], 
        [1, 2, 3],
        [0, 2, 3]
    ])
    
    # Test rendering
    success = render_smplx_mesh(
        vertices, faces, 
        output_path="./test_render.png",
        renderer_type='both',
        camera_position='iso'
    )
    
    if success:
        print("✅ Test render successful: test_render.png")
    else:
        print("❌ Test render failed")