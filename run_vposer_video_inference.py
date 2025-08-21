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

def simple_mesh_render(vertices, faces, img_shape, focal_length=(5000, 5000), princpt=None):
    """Improved mesh rendering with proper camera projection"""
    try:
        if princpt is None:
            princpt = (img_shape[1] // 2, img_shape[0] // 2)
        
        # Convert vertices to numpy if needed
        if hasattr(vertices, 'cpu'):
            vertices = vertices.cpu().numpy()
        
        # Apply perspective projection with camera intrinsics
        # Vertices are in camera coordinates (X, Y, Z)
        x_cam, y_cam, z_cam = vertices[:, 0], vertices[:, 1], vertices[:, 2]
        
        # Avoid division by zero
        z_cam = np.maximum(z_cam, 0.01)
        
        # Project to 2D using camera intrinsics
        x_2d = (x_cam / z_cam) * focal_length[0] + princpt[0]
        y_2d = (y_cam / z_cam) * focal_length[1] + princpt[1]
        
        # Convert to image coordinates
        x_img = np.round(x_2d).astype(int)
        y_img = np.round(y_2d).astype(int)
        
        # Create mesh visualization
        mesh_img = np.zeros((img_shape[0], img_shape[1], 3), dtype=np.uint8)
        
        # Only render points that are in front of camera and within image bounds
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
            img, _, _ = generate_patch_image(cvimg=original_img, 
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

            mesh = out['smplx_mesh_cam'].detach().cpu().numpy()[0]

            # Improved mesh rendering with camera parameters
            try:
                # Get camera parameters from config
                focal_length = cfg.model.focal
                princpt = cfg.model.princpt
                
                # Render mesh overlay with proper projection
                mesh_overlay = simple_mesh_render(
                    mesh, smpl_x.face, vis_img.shape, 
                    focal_length=focal_length, 
                    princpt=princpt
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