#!/usr/bin/env python3
"""
Create a better mesh wireframe visualization for Mac compatibility
"""

import os
import cv2
import numpy as np

def perspective_projection_robust(vertices_3d, camera_params):
    """
    Project 3D vertices to 2D using camera parameters
    """
    focal = camera_params['focal']
    princpt = camera_params['princpt']
    
    vertices_2d = vertices_3d.copy()
    
    # Avoid division by zero
    z_mask = vertices_3d[:, 2] > 0.001
    vertices_2d[z_mask, 0] = vertices_3d[z_mask, 0] * focal[0] / vertices_3d[z_mask, 2] + princpt[0]
    vertices_2d[z_mask, 1] = vertices_3d[z_mask, 1] * focal[1] / vertices_3d[z_mask, 2] + princpt[1]
    
    return vertices_2d, z_mask

def draw_mesh_wireframe(img, vertices_3d, faces, camera_params, color=(0, 255, 255), thickness=1):
    """
    Draw mesh wireframe on image with improved visibility
    """
    img_height, img_width = img.shape[:2]
    
    # Project vertices to 2D
    vertices_2d, valid_mask = perspective_projection_robust(vertices_3d, camera_params)
    
    # Convert to integer coordinates
    vertices_2d_int = vertices_2d.astype(np.int32)
    
    # Draw faces as wireframe
    faces_drawn = 0
    total_faces = len(faces)
    
    for face in faces:
        # Get the three vertices of the face
        v1_idx, v2_idx, v3_idx = face[0], face[1], face[2]
        
        # Check if all vertices are valid (in front of camera)
        if not (valid_mask[v1_idx] and valid_mask[v2_idx] and valid_mask[v3_idx]):
            continue
            
        v1 = vertices_2d_int[v1_idx]
        v2 = vertices_2d_int[v2_idx] 
        v3 = vertices_2d_int[v3_idx]
        
        # Check if vertices are within image bounds (with some margin)
        margin = 50
        if (all(-margin < pt[0] < img_width + margin and -margin < pt[1] < img_height + margin 
               for pt in [v1, v2, v3])):
            
            # Draw triangle edges
            cv2.line(img, tuple(v1), tuple(v2), color, thickness)
            cv2.line(img, tuple(v2), tuple(v3), color, thickness)  
            cv2.line(img, tuple(v3), tuple(v1), color, thickness)
            faces_drawn += 1
    
    print(f"Drew {faces_drawn} faces out of {total_faces} total faces")
    return img

def draw_mesh_points(img, vertices_3d, camera_params, color=(255, 0, 0), radius=2):
    """
    Draw mesh vertices as points for debugging
    """
    img_height, img_width = img.shape[:2]
    
    # Project vertices to 2D
    vertices_2d, valid_mask = perspective_projection_robust(vertices_3d, camera_params)
    
    # Convert to integer coordinates
    vertices_2d_int = vertices_2d.astype(np.int32)
    
    points_drawn = 0
    
    for i, (valid, point) in enumerate(zip(valid_mask, vertices_2d_int)):
        if valid and 0 <= point[0] < img_width and 0 <= point[1] < img_height:
            cv2.circle(img, tuple(point), radius, color, -1)
            points_drawn += 1
    
    print(f"Drew {points_drawn} vertex points out of {len(vertices_3d)} total vertices")
    return img

def render_mesh_improved(img, vertices, faces, camera_params):
    """
    Improved mesh rendering with better Mac compatibility
    """
    print(f"Rendering mesh: {len(vertices)} vertices, {len(faces)} faces")
    print(f"Vertex range: [{np.min(vertices):.3f}, {np.max(vertices):.3f}]")
    print(f"Face range: [{np.min(faces):.0f}, {np.max(faces):.0f}]")
    print(f"Camera params: focal={camera_params['focal']}, princpt={camera_params['princpt']}")
    
    # Create a copy of the image
    result_img = img.copy()
    
    # Method 1: Draw wireframe
    try:
        result_img = draw_mesh_wireframe(result_img, vertices, faces, camera_params, 
                                       color=(0, 255, 255), thickness=1)
    except Exception as e:
        print(f"Wireframe rendering failed: {e}")
    
    # Method 2: Draw key points (subsample for performance)
    try:
        step = max(1, len(vertices) // 500)  # Draw ~500 points max
        sampled_vertices = vertices[::step]
        result_img = draw_mesh_points(result_img, sampled_vertices, camera_params,
                                    color=(255, 0, 0), radius=1)
    except Exception as e:
        print(f"Point rendering failed: {e}")
    
    return result_img

if __name__ == "__main__":
    # Test with an actual output frame
    import sys
    sys.path.append('/Users/vincent.lordier/Work/SMPLest-X')
    
    # Load a sample mesh file
    sample_frame = "/Users/vincent.lordier/Work/SMPLest-X/demo/output_frames/1349093_720p/000010.jpg"
    if os.path.exists(sample_frame):
        img = cv2.imread(sample_frame)
        if img is not None:
            print(f"Loaded test image: {img.shape}")
        else:
            print("Could not load test image")
    else:
        print("Test image not found")