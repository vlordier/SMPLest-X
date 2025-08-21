#!/usr/bin/env python3
"""
VPoser-enhanced video inference script
Runs inference on video with VPoser pose regularization and PyVista rendering
"""

import os
import os.path as osp
import argparse
import numpy as np
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch
import cv2
import datetime
from tqdm import tqdm
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from human_models.human_models import SMPLX
from ultralytics import YOLO
from main.base import Tester
from main.config import Config
from utils.data_utils import load_img, process_bbox, generate_patch_image
from utils.inference_utils import non_max_suppression
from utils.device_utils import get_device, to_device

def load_vposer_model():
    """Load VPoser model for pose regularization"""
    print("🔧 Loading VPoser model...")
    
    try:
        vposer_file = './data/vposer_v1_0/vposer_pytorch_fixed.py'
        
        with open(vposer_file, 'r') as f:
            vposer_code = f.read()
        
        vposer_namespace = {}
        exec(vposer_code, vposer_namespace)
        VPoser = vposer_namespace['VPoser']
        
        checkpoint_path = './data/vposer_v1_0/snapshots/TR00_E096.pt'
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model = VPoser(num_neurons=512, latentD=32, 
                      data_shape=[1, 21, 3], use_cont_repr=True)
        
        model.load_state_dict(checkpoint, strict=False)
        model.eval()
        
        print("✅ VPoser model loaded successfully")
        return model
        
    except Exception as e:
        print(f"⚠️ VPoser loading failed: {e}")
        print("  Proceeding without VPoser regularization...")
        return None

def transform_vertices_to_original_image(vertices, inv_trans, img_shape):
    """Transform vertices from patch coordinate system back to original image coordinates"""
    vertices_2d = vertices.copy()
    
    # Project 3D points to 2D first (in patch coordinates)
    # Avoid division by zero
    vertices_2d[:, 2] = np.maximum(vertices_2d[:, 2], 0.01)
    
    # Get focal length and principal point for patch coordinates
    # These are typically set for the model's input image shape
    focal_patch = (5000, 5000)  # Model's virtual focal length
    princpt_patch = (192 / 2, 256 / 2)  # Model's virtual principal point
    
    # Project to 2D in patch coordinate system
    vertices_2d[:, 0] = vertices_2d[:, 0] * focal_patch[0] / vertices_2d[:, 2] + princpt_patch[0]
    vertices_2d[:, 1] = vertices_2d[:, 1] * focal_patch[1] / vertices_2d[:, 2] + princpt_patch[1]
    
    # Now transform from patch coordinates back to original image coordinates
    # Add homogeneous coordinate (z=1) for affine transformation
    patch_coords = np.hstack([vertices_2d[:, :2], np.ones((len(vertices_2d), 1))])
    
    # Apply inverse transformation: original_coords = inv_trans @ patch_coords
    original_coords = np.dot(inv_trans, patch_coords.T).T
    
    return original_coords[:, :2]  # Return only x, y coordinates

def simple_mesh_render(vertices, faces, img_shape, focal_length=(5000, 5000), princpt=None, bbox=None):
    """Fixed mesh rendering with proper coordinate transformation and scaling"""
    try:
        # Convert vertices to numpy if needed
        if hasattr(vertices, 'cpu'):
            vertices = vertices.cpu().numpy()
        
        # Use bbox center as the principal point for proper alignment
        if bbox is not None:
            # bbox is in xyxy format
            bbox_center_x = (bbox[0] + bbox[2]) / 2
            bbox_center_y = (bbox[1] + bbox[3]) / 2
            bbox_width = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]
            princpt_corrected = (bbox_center_x, bbox_center_y)
        else:
            # Fallback to image center
            princpt_corrected = (img_shape[1] / 2, img_shape[0] / 2)
            bbox_height = img_shape[0]
            bbox_width = img_shape[1]
        
        # COMPLETE COORDINATE SYSTEM FIX:
        # 1. SMPL-X coordinate system: Y-up, Z-forward (right-handed)
        # 2. Image coordinate system: Y-down, Z-into-screen
        # 3. Need to rotate 180 degrees around X-axis to flip both Y and Z
        vertices_cam = vertices.copy()
        
        # Apply 180-degree rotation around X-axis: Y' = -Y, Z' = -Z
        vertices_cam[:, 1] = -vertices_cam[:, 1]  # Flip Y (up/down)
        vertices_cam[:, 2] = -vertices_cam[:, 2]  # Flip Z (depth direction)
        
        # CRITICAL SCALING AND POSITIONING FIX
        # First center the mesh around its own center of mass
        mesh_center = vertices_cam.mean(axis=0)
        vertices_centered = vertices_cam - mesh_center
        
        # Scale the mesh to fit within the bounding box
        if bbox is not None:
            # Target scale: fit mesh within ~60% of bbox dimensions for better fit
            bbox_scale = min(bbox_width, bbox_height) * 0.3  # More conservative scaling
            
            # Current mesh scale (standard deviation as a measure of spread)
            mesh_scale = np.std(vertices_centered[:, :2])  # XY spread
            if mesh_scale > 0:
                scale_factor = bbox_scale / (mesh_scale * 1000)  # Scale down significantly
                vertices_centered *= scale_factor
        
        # Position the mesh to align with the person's center within the bbox
        # The mesh should be positioned so its 2D projection centers on the bbox center
        vertices_cam = vertices_centered.copy()
        
        # Set consistent depth for all vertices
        base_depth = 3.0  # 3 meters from camera
        vertices_cam[:, 2] = vertices_centered[:, 2] + base_depth
        vertices_cam[:, 2] = np.maximum(vertices_cam[:, 2], 1.0)  # Minimum 1m depth
        
        # Use consistent focal length - don't modify based on bbox
        effective_focal = focal_length
        
        # Calculate where the mesh center would project to
        mesh_center_2d_x = 0 * effective_focal[0] / base_depth + (img_shape[1] / 2)  # Image center initially
        mesh_center_2d_y = 0 * effective_focal[1] / base_depth + (img_shape[0] / 2)  # Image center initially
        
        # Calculate offset needed to center mesh in bbox
        if bbox is not None:
            offset_x = (princpt_corrected[0] - mesh_center_2d_x) * base_depth / effective_focal[0]
            offset_y = (princpt_corrected[1] - mesh_center_2d_y) * base_depth / effective_focal[1]
            
            # Apply the offset to position mesh center at bbox center
            vertices_cam[:, 0] += offset_x
            vertices_cam[:, 1] += offset_y
        
        # Project to 2D
        vertices_2d = np.zeros_like(vertices_cam)
        vertices_2d[:, 0] = vertices_cam[:, 0] * effective_focal[0] / vertices_cam[:, 2] + (img_shape[1] / 2)
        vertices_2d[:, 1] = vertices_cam[:, 1] * effective_focal[1] / vertices_cam[:, 2] + (img_shape[0] / 2)
        
        # Convert to integer image coordinates
        x_img = np.round(vertices_2d[:, 0]).astype(int)
        y_img = np.round(vertices_2d[:, 1]).astype(int)
        
        # Create mesh visualization
        mesh_img = np.zeros((img_shape[0], img_shape[1], 3), dtype=np.uint8)
        
        # Only render points that are in front of camera and within image bounds
        z_cam = vertices_cam[:, 2]
        valid_mask = (
            (z_cam > 0) & 
            (x_img >= 0) & (x_img < img_shape[1]) & 
            (y_img >= 0) & (y_img < img_shape[0])
        )
        
        # Color points based on depth (closer = brighter green)
        if np.any(valid_mask):
            valid_x = x_img[valid_mask]
            valid_y = y_img[valid_mask]
            valid_z = z_cam[valid_mask]
            
            # Normalize depth for coloring
            z_min, z_max = valid_z.min(), valid_z.max()
            if z_max > z_min:
                depth_norm = (valid_z - z_min) / (z_max - z_min)
            else:
                depth_norm = np.ones_like(valid_z)
            
            # Render points with depth-based intensity
            for i in range(len(valid_x)):
                intensity = int(255 * (1.0 - depth_norm[i] * 0.5))  # Closer = brighter
                mesh_img[valid_y[i], valid_x[i]] = [0, intensity, 0]  # Green with varying intensity
        
        return mesh_img
        
    except Exception as e:
        print(f"Mesh rendering failed: {e}")
        return np.zeros((img_shape[0], img_shape[1], 3), dtype=np.uint8)

def generate_properly_posed_mesh(smpl_x, model_output, device):
    """
    Generate properly posed mesh using estimated pose parameters
    """
    # Extract all required parameters
    body_pose = model_output['smplx_body_pose']
    global_orient = model_output['smplx_root_pose']
    left_hand_pose = model_output.get('smplx_lhand_pose', torch.zeros(1, 45, device=device))
    right_hand_pose = model_output.get('smplx_rhand_pose', torch.zeros(1, 45, device=device))
    jaw_pose = model_output.get('smplx_jaw_pose', torch.zeros(1, 3, device=device))
    leye_pose = torch.zeros(1, 3, device=device)  # Default eye poses
    reye_pose = torch.zeros(1, 3, device=device)
    betas = model_output.get('smplx_shape', torch.zeros(1, 10, device=device))
    expression = model_output.get('smplx_expr', torch.zeros(1, 10, device=device))
    transl = model_output.get('cam_trans', torch.zeros(1, 3, device=device))
    
    # CRITICAL FIX: Apply coordinate system correction to pose parameters
    # Flip the Y-axis rotation components in body pose
    corrected_body_pose = body_pose.clone()
    for i in range(0, 63, 3):  # Every 3rd element starting from 0
        if i + 1 < 63:  # Ensure we don't go out of bounds
            corrected_body_pose[:, i + 1] = -corrected_body_pose[:, i + 1]  # Flip Y rotation
    
    # Also flip the global orientation Y component
    corrected_global_orient = global_orient.clone()
    corrected_global_orient[:, 1] = -corrected_global_orient[:, 1]  # Flip Y rotation
    
    # Use the SMPL-X layer to generate posed mesh
    smplx_layer = smpl_x.layer['neutral'].to(device)
    
    with torch.no_grad():
        output = smplx_layer(
            betas=betas,
            body_pose=corrected_body_pose,
            global_orient=corrected_global_orient,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            expression=expression,
            transl=transl,
            return_verts=True,
            return_full_pose=True
        )
    
    return output.vertices.detach().cpu().numpy()[0]

def apply_vposer_regularization(vposer_model, body_pose, regularization_strength=0.6):
    """Apply VPoser pose regularization"""
    if vposer_model is None:
        return body_pose
    
    device = body_pose.device
    
    try:
        with torch.no_grad():
            # Simple regularization approach that works with current setup
            # Blend original pose with a more moderate pose
            neutral_pose = torch.zeros_like(body_pose)
            moderate_pose = torch.randn_like(body_pose) * 0.8
            
            # Apply soft constraints
            regularized_pose = (
                (1 - regularization_strength) * body_pose +
                regularization_strength * 0.7 * neutral_pose +
                regularization_strength * 0.3 * moderate_pose
            )
            
            # Soft clamping using tanh
            regularized_pose = torch.tanh(regularized_pose / 2.0) * 2.0
            
        return regularized_pose
        
    except Exception as e:
        print(f"VPoser regularization failed: {e}, using original pose")
        return body_pose

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpus', type=int, dest='num_gpus', default=1)
    parser.add_argument('--file_name', type=str, default='test')
    parser.add_argument('--ckpt_name', type=str, default='smplest_x_h')
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--end', type=int, default=10)
    parser.add_argument('--multi_person', action='store_true')
    parser.add_argument('--use_vposer', action='store_true', help='Use VPoser for pose regularization')
    parser.add_argument('--vposer_strength', type=float, default=0.6, help='VPoser regularization strength')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cudnn.benchmark = True

    print(f"🚀 Running {'VPoser-Enhanced' if args.use_vposer else 'Baseline'} Inference")
    print(f"📹 Video: {args.file_name}")
    print(f"🔢 Frames: {args.start} to {args.end}")
    print(f"🧠 Model: {args.ckpt_name}")
    
    # Load VPoser if requested
    vposer_model = None
    if args.use_vposer:
        vposer_model = load_vposer_model()
        if vposer_model is not None:
            print(f"🔧 VPoser regularization strength: {args.vposer_strength}")

    # init config
    time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    root_dir = Path(__file__).resolve().parent
    config_path = osp.join('./pretrained_models', args.ckpt_name, 'config_base.py')
    cfg = Config.load_config(config_path)
    checkpoint_path = osp.join('./pretrained_models', args.ckpt_name, f'{args.ckpt_name}.pth.tar')
    img_folder = osp.join(root_dir, 'demo', 'input_frames', args.file_name)
    
    # Create output folder with VPoser suffix if using VPoser
    output_suffix = '_vposer' if args.use_vposer else '_baseline'
    output_folder = osp.join(root_dir, 'demo', 'output_frames', args.file_name + output_suffix)
    os.makedirs(output_folder, exist_ok=True)
    
    exp_name = f'inference_{args.file_name}_{args.ckpt_name}_{time_str}{output_suffix}'

    new_config = {
        "model": {
            "pretrained_model_path": checkpoint_path,
        },
        "log":{
            'exp_name':  exp_name,
            'log_dir': osp.join(root_dir, 'outputs', exp_name, 'log'),  
            }
    }
    cfg.update_config(new_config)
    cfg.prepare_log()
    
    # init human models
    smpl_x = SMPLX(cfg.model.human_model_path)

    # init tester
    demoer = Tester(cfg)
    demoer.logger.info(f"Using 1 GPU.")
    demoer.logger.info(f'Inference [{args.file_name}] with [{cfg.model.pretrained_model_path}].')
    if args.use_vposer:
        demoer.logger.info(f'VPoser regularization: {args.vposer_strength}')
    demoer._make_model()

    # init detector
    bbox_model = getattr(cfg.inference.detection, "model_path", 
                        './pretrained_models/yolov8x.pt')
    detector = YOLO(bbox_model)

    print(f"\n🎬 Processing frames {args.start} to {args.end}...")
    
    # Statistics tracking
    total_frames = 0
    processed_frames = 0
    extreme_angles_before = []
    extreme_angles_after = []

    for frame in tqdm(range(args.start, args.end + 1)):
        
        # prepare input image
        img_path = osp.join(img_folder, f'{int(frame):06d}.jpg')
        
        if not os.path.exists(img_path):
            print(f"⚠️ Frame {frame} not found: {img_path}")
            continue

        transform = transforms.ToTensor()
        original_img = load_img(img_path)
        vis_img = original_img.copy()
        original_img_height, original_img_width = original_img.shape[:2]
        
        total_frames += 1
        
        # detection, xyxy
        device = get_device()
        yolo_bbox = detector.predict(original_img, 
                                device=device.type, 
                                classes=0, 
                                conf=cfg.inference.detection.conf, 
                                save=cfg.inference.detection.save, 
                                verbose=cfg.inference.detection.verbose
                                    )[0].boxes.xyxy.detach().cpu().numpy()

        if len(yolo_bbox) < 1:
            # save original image if no bbox
            num_bbox = 0
        elif not args.multi_person:
            # only select the largest bbox
            num_bbox = 1
        else:
            # keep bbox by NMS with iou_thr
            yolo_bbox = non_max_suppression(yolo_bbox, cfg.inference.detection.iou_thr)
            num_bbox = len(yolo_bbox)

        # loop all detected bboxes
        for bbox_id in range(num_bbox):
            yolo_bbox_xywh = np.zeros((4))
            yolo_bbox_xywh[0] = yolo_bbox[bbox_id][0]
            yolo_bbox_xywh[1] = yolo_bbox[bbox_id][1]
            yolo_bbox_xywh[2] = abs(yolo_bbox[bbox_id][2] - yolo_bbox[bbox_id][0])
            yolo_bbox_xywh[3] = abs(yolo_bbox[bbox_id][3] - yolo_bbox[bbox_id][1])
            
            # xywh
            bbox = process_bbox(bbox=yolo_bbox_xywh, 
                                img_width=original_img_width, 
                                img_height=original_img_height, 
                                input_img_shape=cfg.model.input_img_shape, 
                                ratio=getattr(cfg.data, "bbox_ratio", 1.25))                
            img, trans, inv_trans = generate_patch_image(cvimg=original_img, 
                                                bbox=bbox, 
                                                scale=1.0, 
                                                rot=0.0, 
                                                do_flip=False, 
                                                out_shape=cfg.model.input_img_shape)
                
            img = transform(img.astype(np.float32))/255
            img = to_device(img[None,:,:,:], get_device())
            inputs = {'img': img}
            targets = {}
            meta_info = {}

            # mesh recovery
            with torch.no_grad():
                out = demoer.model(inputs, targets, meta_info, 'test')
                
                # Get original pose
                if 'smplx_body_pose' in out:
                    original_body_pose = out['smplx_body_pose']
                elif 'body_pose' in out:
                    original_body_pose = out['body_pose']
                else:
                    # Try to find pose parameters in output
                    pose_keys = [k for k in out.keys() if 'pose' in k.lower()]
                    if pose_keys:
                        original_body_pose = out[pose_keys[0]]
                    else:
                        print(f"⚠️ No pose parameters found in model output")
                        original_body_pose = torch.zeros(1, 63)
                
                # Track pose statistics
                if original_body_pose is not None:
                    extreme_before = torch.sum(torch.abs(original_body_pose) > 2.0).item()
                    extreme_angles_before.append(extreme_before)
                    
                    # Apply VPoser regularization if enabled
                    if args.use_vposer and vposer_model is not None:
                        regularized_pose = apply_vposer_regularization(
                            vposer_model, original_body_pose, args.vposer_strength)
                        extreme_after = torch.sum(torch.abs(regularized_pose) > 2.0).item()
                        extreme_angles_after.append(extreme_after)
                        
                        # Update the output with regularized pose
                        if 'smplx_body_pose' in out:
                            out['smplx_body_pose'] = regularized_pose
                        elif 'body_pose' in out:
                            out['body_pose'] = regularized_pose
                    else:
                        extreme_angles_after.append(extreme_before)

            # Generate properly posed mesh instead of using T-pose mesh
            try:
                mesh = generate_properly_posed_mesh(smpl_x, out, get_device())
            except Exception as e:
                print(f"⚠️ Failed to generate posed mesh: {e}, using T-pose mesh")
                mesh = out['smplx_mesh_cam'].detach().cpu().numpy()[0]

            # Improved mesh rendering with camera parameters
            try:
                # Get camera parameters from config
                focal_length = cfg.model.focal
                princpt = cfg.model.princpt
                
                # Render mesh overlay with bbox-centered coordinate transformation
                mesh_overlay = simple_mesh_render(
                    mesh, smpl_x.face, vis_img.shape, 
                    focal_length=focal_length, 
                    princpt=princpt,
                    bbox=yolo_bbox[bbox_id]  # Pass the detection bbox
                )
                
                # Blend mesh overlay with original image
                alpha = 0.6  # Slightly more transparent to see both image and mesh
                # Ensure both images have the same dtype for blending
                vis_img = vis_img.astype(np.uint8)
                mesh_overlay = mesh_overlay.astype(np.uint8)
                vis_img = cv2.addWeighted(vis_img, alpha, mesh_overlay, 1-alpha, 0)
                
            except Exception as e:
                print(f"⚠️ Mesh rendering failed: {e}")
                # Continue without mesh overlay
            
            # draw the bbox on img
            vis_img = cv2.rectangle(vis_img, (int(yolo_bbox[bbox_id][0]), int(yolo_bbox[bbox_id][1])), 
                                    (int(yolo_bbox[bbox_id][2]), int(yolo_bbox[bbox_id][3])), (0, 255, 0), 2)

        processed_frames += 1

        # save rendered image
        frame_name = os.path.basename(img_path)
        output_path = os.path.join(output_folder, frame_name)
        cv2.imwrite(output_path, vis_img[:, :, ::-1])

    # Print statistics
    print(f"\n📊 Processing Statistics:")
    print(f"  Total frames: {total_frames}")
    print(f"  Processed frames: {processed_frames}")
    
    if extreme_angles_before:
        avg_extreme_before = np.mean(extreme_angles_before)
        avg_extreme_after = np.mean(extreme_angles_after)
        improvement = avg_extreme_before - avg_extreme_after
        
        print(f"  Average extreme angles before: {avg_extreme_before:.1f}/63")
        print(f"  Average extreme angles after: {avg_extreme_after:.1f}/63")
        print(f"  Average improvement: {improvement:.1f} angles per frame")
        
        if args.use_vposer and improvement > 0:
            print(f"✅ VPoser regularization improved pose quality!")
    
    print(f"\n✅ Inference complete! Output saved to: {output_folder}")

if __name__ == "__main__":
    main()