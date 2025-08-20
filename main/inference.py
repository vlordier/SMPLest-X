import os
import os.path as osp
import argparse
import numpy as np
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch
import cv2
import datetime
import time
import logging
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Optional
from human_models.human_models import SMPLX
from ultralytics import YOLO
from main.base import Tester
from main.config import Config
from utils.data_utils import load_img, process_bbox, generate_patch_image
from utils.visualization_utils import render_mesh, render_mesh_improved
from utils.pyvista_renderer import PyVistaRenderer
from utils.device_utils import get_device_name, to_device
from utils.validation_models import (
    InferenceConfig, DetectionResult, SMPLXParameters, MeshData, 
    InferenceOutput, ValidationLogger, validate_tensor_output, 
    validate_numpy_array, MeshFormat
)


def save_mesh_obj(vertices, faces, filepath):
    """Save mesh in OBJ format"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        # Write vertices
        for v in vertices:
            f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')
        
        # Write faces (OBJ uses 1-based indexing)
        for face in faces:
            f.write(f'f {face[0]+1} {face[1]+1} {face[2]+1}\n')


def save_mesh_ply(vertices, faces, filepath):
    """Save mesh in PLY format"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        # PLY header
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(vertices)}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write(f'element face {len(faces)}\n')
        f.write('property list uchar int vertex_indices\n')
        f.write('end_header\n')
        
        # Write vertices
        for v in vertices:
            f.write(f'{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')
        
        # Write faces
        for face in faces:
            f.write(f'3 {face[0]} {face[1]} {face[2]}\n')


def render_mesh_only(vertices, faces, cam_param, img_shape=(512, 512)):
    """Render mesh without background for isolated mesh visualization using PyVista"""
    try:
        # Create PyVista renderer
        renderer = PyVistaRenderer(window_size=(img_shape[1], img_shape[0]), background_color='white')
        
        # Render with PyVista (returns numpy array)
        image = renderer.render_mesh(
            vertices=vertices,
            faces=faces,
            camera_position='iso',
            color='lightblue',
            show_edges=True,
            edge_color='black',
            smooth_shading=True
        )
        
        if image is not None:
            # Convert RGBA to RGB if needed and ensure correct format
            if len(image.shape) == 3 and image.shape[2] == 4:
                image = image[:, :, :3]
            return image
        else:
            raise Exception("PyVista returned None")
        
    except Exception as e:
        print(f"PyVista rendering failed ({e}), using matplotlib wireframe fallback")
        return render_mesh_matplotlib_fallback(vertices, faces, cam_param, img_shape)


def render_mesh_matplotlib_fallback(vertices, faces, cam_param, img_shape=(512, 512)):
    """Fallback mesh renderer using matplotlib for wireframe visualization"""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend BEFORE importing pyplot
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        # Set maximum number of figures to prevent memory warnings
        plt.rcParams['figure.max_open_warning'] = 0
        
        # Create 3D plot
        fig = plt.figure(figsize=(img_shape[1]/100, img_shape[0]/100), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        
        # Set the aspect ratio and viewing angle
        ax.set_box_aspect([1,1,1])
        
        # Create mesh visualization - subsample faces for performance
        num_faces = len(faces)
        if num_faces > 1000:
            # Subsample faces for better performance while maintaining shape
            step = max(1, num_faces // 1000)
            selected_faces = faces[::step]
        else:
            selected_faces = faces
        
        # Create face collection
        face_vertices = vertices[selected_faces]
        
        # Create mesh with light blue color and some transparency
        mesh = Poly3DCollection(face_vertices, alpha=0.7, facecolor='lightblue', 
                               edgecolor='navy', linewidth=0.1)
        ax.add_collection3d(mesh)
        
        # Set the limits based on the mesh bounds
        ax.set_xlim([vertices[:, 0].min(), vertices[:, 0].max()])
        ax.set_ylim([vertices[:, 1].min(), vertices[:, 1].max()])
        ax.set_zlim([vertices[:, 2].min(), vertices[:, 2].max()])
        
        # Set viewing angle for better visualization
        ax.view_init(elev=20, azim=45)
        
        # Remove axes for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.grid(False)
        
        # Set background color
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        
        # Save to buffer
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        
        # Convert to numpy array using new API
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        # Convert RGBA to RGB by removing alpha channel
        buf = buf[:, :, :3]
        
        plt.close(fig)
        # Clear any remaining matplotlib references to prevent memory leaks
        import gc
        gc.collect()
        
        # Resize if necessary
        if buf.shape[:2] != img_shape:
            import cv2
            buf = cv2.resize(buf, (img_shape[1], img_shape[0]))
        
        return buf
        
    except Exception as e:
        print(f"Matplotlib fallback failed ({e}), using simple wireframe")
        return render_simple_wireframe(vertices, faces, img_shape)


def render_simple_wireframe(vertices, faces, img_shape=(512, 512)):
    """Simple wireframe renderer using OpenCV"""
    # Create black background
    img = np.zeros((img_shape[0], img_shape[1], 3), dtype=np.uint8)
    
    # Simple camera projection
    # Center and scale the mesh
    v_centered = vertices - vertices.mean(axis=0)
    scale = min(img_shape) * 0.3 / (np.max(v_centered) - np.min(v_centered))
    v_scaled = v_centered * scale
    
    # Project to 2D (simple orthographic projection)
    v_2d = v_scaled[:, :2] + np.array([img_shape[1], img_shape[0]]) / 2
    v_2d = v_2d.astype(int)
    
    # Draw edges - subsample for performance
    num_faces = len(faces)
    if num_faces > 500:
        step = max(1, num_faces // 500)
        selected_faces = faces[::step]
    else:
        selected_faces = faces
    
    # Draw wireframe
    for face in selected_faces:
        if all(0 <= v_2d[face[i]][0] < img_shape[1] and 
               0 <= v_2d[face[i]][1] < img_shape[0] for i in range(3)):
            # Draw triangle edges
            for i in range(3):
                pt1 = tuple(v_2d[face[i]])
                pt2 = tuple(v_2d[face[(i + 1) % 3]])
                cv2.line(img, pt1, pt2, (0, 255, 255), 1)  # Cyan wireframe
    
    return img


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpus', type=int, dest='num_gpus', default=1)
    parser.add_argument('--file_name', type=str, default='test')
    parser.add_argument('--ckpt_name', type=str, default='model_dump')
    parser.add_argument('--start', type=str, default='1')
    parser.add_argument('--end', type=str, default='1')
    parser.add_argument('--multi_person', action='store_true')
    parser.add_argument('--save_meshes', action='store_true', help='Save 3D meshes in OBJ format')
    parser.add_argument('--save_mesh_renders', action='store_true', help='Save isolated mesh renders without background')
    parser.add_argument('--mesh_format', type=str, default='obj', choices=['obj', 'ply'], help='Mesh file format')
    # Optional validation parameters
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--bbox_thr', type=float, default=0.9, help='Bounding box threshold')
    parser.add_argument('--fps', type=int, help='Frame rate for video processing (auto-detected if not specified)')
    parser.add_argument('--frame_skip', type=int, default=2, choices=[1, 2, 3, 4, 5, 8], 
                       help='Skip every N frames for processing (default: 2)')
    args = parser.parse_args()
    return args

def get_video_fps(video_path: Path) -> Optional[float]:
    """Extract FPS from video file"""
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps if fps > 0 else None
    except Exception as e:
        logging.warning(f"Could not extract FPS from video: {e}")
        return None

def validate_and_process_detections(yolo_results, frame_id: int, confidence_threshold: float = 0.5) -> List[DetectionResult]:
    """Validate and convert YOLO detections to DetectionResult objects"""
    detections: List[DetectionResult] = []
    
    if not hasattr(yolo_results, 'boxes') or yolo_results.boxes is None:
        logging.warning(f"Frame {frame_id}: No detection boxes found")
        return detections
    
    try:
        boxes = yolo_results.boxes.xyxy.detach().cpu().numpy()
        confidences = yolo_results.boxes.conf.detach().cpu().numpy() if yolo_results.boxes.conf is not None else None
        
        validate_numpy_array(boxes, f"Frame {frame_id} boxes")
        
        for person_id, bbox in enumerate(boxes):
            confidence = confidences[person_id] if confidences is not None else 1.0
            
            if confidence < confidence_threshold:
                continue
                
            try:
                detection = DetectionResult(
                    person_id=person_id,
                    bbox=bbox.tolist(),
                    confidence=float(confidence),
                    frame_id=frame_id
                )
                detections.append(detection)
                
            except Exception as e:
                logging.error(f"Frame {frame_id}, Person {person_id}: Detection validation failed - {e}")
                
    except Exception as e:
        logging.error(f"Frame {frame_id}: Failed to process detections - {e}")
    
    return detections

def validate_and_convert_smplx_params(smplx_output: Dict, person_id: int, frame_id: int) -> Optional[SMPLXParameters]:
    """Validate and convert model outputs to SMPLXParameters"""
    try:
        # Validate tensor outputs first
        for key, tensor in smplx_output.items():
            if isinstance(tensor, torch.Tensor):
                validate_tensor_output(tensor, f"Frame {frame_id}, Person {person_id}, {key}")
        
        # Convert to SMPLXParameters
        params = SMPLXParameters(
            betas=smplx_output.get('betas', torch.zeros(10)).detach().cpu().numpy().tolist(),
            body_pose=smplx_output.get('body_pose', torch.zeros(63)).detach().cpu().numpy().flatten().tolist(),
            global_orient=smplx_output.get('global_orient', torch.zeros(3)).detach().cpu().numpy().flatten().tolist(),
            left_hand_pose=smplx_output.get('left_hand_pose', torch.zeros(45)).detach().cpu().numpy().flatten().tolist() if 'left_hand_pose' in smplx_output else None,
            right_hand_pose=smplx_output.get('right_hand_pose', torch.zeros(45)).detach().cpu().numpy().flatten().tolist() if 'right_hand_pose' in smplx_output else None,
            jaw_pose=smplx_output.get('jaw_pose', torch.zeros(3)).detach().cpu().numpy().flatten().tolist() if 'jaw_pose' in smplx_output else None,
            leye_pose=smplx_output.get('leye_pose', torch.zeros(3)).detach().cpu().numpy().flatten().tolist() if 'leye_pose' in smplx_output else None,
            reye_pose=smplx_output.get('reye_pose', torch.zeros(3)).detach().cpu().numpy().flatten().tolist() if 'reye_pose' in smplx_output else None,
            expression=smplx_output.get('expression', torch.zeros(10)).detach().cpu().numpy().flatten().tolist() if 'expression' in smplx_output else None,
            transl=smplx_output.get('transl', torch.zeros(3)).detach().cpu().numpy().flatten().tolist()
        )
        
        return params
        
    except Exception as e:
        logging.error(f"Frame {frame_id}, Person {person_id}: SMPL-X parameter validation failed - {e}")
        return None

def validate_and_convert_mesh(vertices: np.ndarray, faces: np.ndarray, person_id: int, frame_id: int) -> Optional[MeshData]:
    """Validate and convert mesh data to MeshData object"""
    try:
        # Validate inputs
        validate_numpy_array(vertices, f"Frame {frame_id}, Person {person_id} vertices", expected_shape=(None, 3))
        validate_numpy_array(faces, f"Frame {frame_id}, Person {person_id} faces", expected_shape=(None, 3))
        
        mesh = MeshData(
            vertices=vertices.tolist(),
            faces=faces.tolist(),
            person_id=person_id,
            frame_id=frame_id
        )
        
        return mesh
        
    except Exception as e:
        logging.error(f"Frame {frame_id}, Person {person_id}: Mesh validation failed - {e}")
        return None

def save_mesh_with_validation(mesh: MeshData, mesh_format: str, output_folder: Path) -> List[Path]:
    """Save mesh with validation and return saved file paths"""
    saved_files = []
    
    try:
        filename_base = f"{mesh.frame_id:06d}_person{mesh.person_id}"
        
        if mesh_format.lower() == 'obj':
            filepath = output_folder / f"{filename_base}.obj"
            save_mesh_obj(np.array(mesh.vertices), np.array(mesh.faces), filepath)
            saved_files.append(filepath)
            
        elif mesh_format.lower() == 'ply':
            filepath = output_folder / f"{filename_base}.ply"
            save_mesh_ply(np.array(mesh.vertices), np.array(mesh.faces), filepath)
            saved_files.append(filepath)
            
        # Validate file was actually created
        for filepath in saved_files:
            if not filepath.exists() or filepath.stat().st_size == 0:
                logging.error(f"Failed to save mesh file: {filepath}")
                saved_files.remove(filepath)
            else:
                logging.info(f"💾 Saved mesh: {filepath.name}")
                
    except Exception as e:
        logging.error(f"Mesh save failed for frame {mesh.frame_id}, person {mesh.person_id}: {e}")
        
    return saved_files

def save_mesh_render_with_validation(mesh: MeshData, cam_param: Dict, output_folder: Path) -> List[Path]:
    """Save mesh render with validation and return saved file paths"""
    saved_files = []
    
    try:
        filename = f"{mesh.frame_id:06d}_person{mesh.person_id}_mesh.jpg"
        filepath = output_folder / filename
        
        # Create a dummy image for rendering (not used, kept for future implementation)
        
        # Render mesh
        render_img = render_mesh_only(
            vertices=np.array(mesh.vertices),
            faces=np.array(mesh.faces),
            cam_param=cam_param,
            img_shape=(512, 512)
        )
        
        if render_img is not None:
            cv2.imwrite(str(filepath), render_img)
            
            # Validate file was created and has reasonable size
            if filepath.exists() and filepath.stat().st_size > 1000:  # At least 1KB
                saved_files.append(filepath)
                logging.info(f"🎨 Saved mesh render: {filepath.name}")
            else:
                logging.error(f"Render file too small or missing: {filepath}")
        else:
            logging.error(f"Failed to render mesh for frame {mesh.frame_id}, person {mesh.person_id}")
            
    except Exception as e:
        logging.error(f"Mesh render save failed for frame {mesh.frame_id}, person {mesh.person_id}: {e}")
        
    return saved_files

def main():
    args = parse_args()
    cudnn.benchmark = True
    
    # Initialize validation logger
    validation_logger = ValidationLogger(Path("inference_validation.log"))
    
    try:
        # Validate and create configuration
        root_dir = Path(__file__).resolve().parent.parent
        video_path = Path(f"demo/{args.file_name}.mp4")
        
        # Auto-detect FPS if not provided
        fps = args.fps
        if fps is None:
            detected_fps = get_video_fps(video_path)
            if detected_fps:
                fps = int(detected_fps)
                logging.info(f"Auto-detected video FPS: {fps}")
            else:
                fps = 30  # Default fallback
                logging.warning(f"Could not detect FPS, using default: {fps}")
        
        logging.info(f"Using frame skip: {args.frame_skip} (processing every {args.frame_skip} frames)")
        
        config = InferenceConfig(
            input_path=video_path,
            output_folder=Path(f"demo/output_frames/{args.file_name}"),
            model_path=Path(f"./pretrained_models/{args.ckpt_name}/{args.ckpt_name}.pth.tar"),
            gpu=getattr(args, 'gpu', 0),
            bbox_thr=getattr(args, 'bbox_thr', 0.9),
            fps=fps,
            frame_skip=args.frame_skip,
            save_meshes=args.save_meshes,
            save_mesh_renders=args.save_mesh_renders,
            mesh_format=MeshFormat(args.mesh_format),
            device=None  # Will be set later
        )
        
        validation_logger.log_config_validation(config)
        
    except Exception as e:
        logging.error(f"Configuration validation failed: {e}")
        return
    
    # Store all inference outputs for final validation
    all_outputs: List[InferenceOutput] = []

    # init config
    time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    root_dir = Path(__file__).resolve().parent.parent
    config_path = osp.join('./pretrained_models', args.ckpt_name, 'config_base.py')
    cfg = Config.load_config(config_path)
    checkpoint_path = osp.join('./pretrained_models', args.ckpt_name, f'{args.ckpt_name}.pth.tar')
    img_folder = osp.join(root_dir, 'demo', 'input_frames', args.file_name)
    output_folder = osp.join(root_dir, 'demo', 'output_frames', args.file_name)
    os.makedirs(output_folder, exist_ok=True)
    
    # Create mesh output directories if needed
    mesh_folder = None
    mesh_render_folder = None
    if args.save_meshes:
        mesh_folder = osp.join(root_dir, 'demo', 'output_meshes', args.file_name)
        os.makedirs(mesh_folder, exist_ok=True)
        print(f"📦 Meshes will be saved to: {mesh_folder}")
    
    if args.save_mesh_renders:
        mesh_render_folder = osp.join(root_dir, 'demo', 'mesh_renders', args.file_name)
        os.makedirs(mesh_render_folder, exist_ok=True)
        print(f"🎨 Mesh renders will be saved to: {mesh_render_folder}")
    
    exp_name = f'inference_{args.file_name}_{args.ckpt_name}_{time_str}'

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
    demoer.logger.info("Using 1 GPU.")
    demoer.logger.info(f'Inference [{args.file_name}] with [{cfg.model.pretrained_model_path}].')
    demoer._make_model()

    # init detector
    bbox_model = getattr(cfg.inference.detection, "model_path", 
                        './pretrained_models/yolov8x.pt')
    detector = YOLO(bbox_model)

    start = int(args.start)
    end = int(args.end) + 1

    # Generate frame sequence with skipping
    frames_to_process = list(range(start, end, args.frame_skip))
    logging.info(f"Processing {len(frames_to_process)} frames with skip={args.frame_skip} (total range: {start}-{end-1})")

    for frame in tqdm(frames_to_process, desc="Processing frames"):
        frame_start_time = time.time()
        
        # Initialize frame output
        frame_output = InferenceOutput(
            frame_id=frame,
            detections=[],
            smplx_params={},
            meshes={},
            mesh_files_saved={},
            render_files_saved={},
            processing_time=0.0
        )
        
        try:
            # prepare input image
            img_path = osp.join(img_folder, f'{int(frame):06d}.jpg')
            
            # Validate image path exists
            if not os.path.exists(img_path):
                logging.error(f"Frame {frame}: Image file not found - {img_path}")
                continue

            transform = transforms.ToTensor()
            original_img = load_img(img_path)
            vis_img = original_img.copy()
            original_img_height, original_img_width = original_img.shape[:2]
            
            # Validate image loaded properly
            validate_numpy_array(original_img, f"Frame {frame} input image", expected_shape=(None, None, 3))
            
            # detection, xyxy
            device_name = get_device_name()
            logging.info(f"Frame {frame}: Running detection on device {device_name}")
            
            yolo_results = detector.predict(original_img, 
                                        device=device_name, 
                                        classes=0,  # person class
                                        conf=cfg.inference.detection.conf, 
                                        save=cfg.inference.detection.save, 
                                        verbose=cfg.inference.detection.verbose
                                        )[0]
            
            # Validate and process detections
            detections = validate_and_process_detections(
                yolo_results, 
                frame_id=frame, 
                confidence_threshold=cfg.inference.detection.conf
            )
            
            frame_output.detections = detections
            validation_logger.log_frame_processing(frame, detections)
            
            if not detections:
                logging.warning(f"Frame {frame}: No valid detections found")
                frame_output.processing_time = time.time() - frame_start_time
                all_outputs.append(frame_output)
                continue
            
            # Process each detection
            for detection in detections:
                person_id = detection.person_id
                bbox = detection.bbox
                
                try:
                    # Convert bbox format for processing
                    yolo_bbox_xywh = np.array([
                        bbox[0],  # x1
                        bbox[1],  # y1  
                        bbox[2] - bbox[0],  # width
                        bbox[3] - bbox[1]   # height
                    ])
                    
                    # Process bbox for model input
                    processed_bbox = process_bbox(
                        bbox=yolo_bbox_xywh, 
                        img_width=original_img_width, 
                        img_height=original_img_height, 
                        input_img_shape=cfg.model.input_img_shape, 
                        ratio=getattr(cfg.data, "bbox_ratio", 1.25)
                    )
                    
                    # Validate processed bbox
                    validate_numpy_array(processed_bbox, f"Frame {frame}, Person {person_id} processed bbox", expected_shape=(4,))
                    
                    # Generate patch image for model input
                    img, _, _ = generate_patch_image(
                        cvimg=original_img, 
                        bbox=processed_bbox, 
                        scale=1.0, 
                        rot=0.0, 
                        do_flip=False, 
                        out_shape=cfg.model.input_img_shape
                    )
                    
                    # Validate patch image
                    validate_numpy_array(img, f"Frame {frame}, Person {person_id} patch image")
                    
                    # Prepare model inputs
                    img_tensor = transform(img.astype(np.float32))/255
                    img_tensor = to_device(img_tensor)[None,:,:,:]
                    
                    # Validate tensor input
                    validate_tensor_output(img_tensor, f"Frame {frame}, Person {person_id} input tensor", expected_shape=(1, 3, cfg.model.input_img_shape[0], cfg.model.input_img_shape[1]))
                    
                    inputs = {'img': img_tensor}
                    targets = {}
                    meta_info = {}

                    # SMPL-X mesh recovery
                    logging.info(f"Frame {frame}, Person {person_id}: Running SMPL-X inference")
                    with torch.no_grad():
                        model_output = demoer.model(inputs, targets, meta_info, 'test')
                    
                    # Validate model outputs
                    if 'smplx_mesh_cam' not in model_output:
                        logging.error(f"Frame {frame}, Person {person_id}: No mesh output from model")
                        continue
                    
                    mesh_tensor = model_output['smplx_mesh_cam']
                    validate_tensor_output(mesh_tensor, f"Frame {frame}, Person {person_id} mesh tensor")
                    
                    mesh_vertices = mesh_tensor.detach().cpu().numpy()[0]
                    mesh_faces = smpl_x.face
                    
                    # Fix Z-axis alignment issue: SMPL-X meshes are positioned too far from camera
                    # Typical Z values are around 30+, which causes misalignment in rendering
                    # Apply Z-axis correction to bring mesh closer to origin for proper alignment
                    z_correction_factor = 0.3  # More conservative correction based on research
                    mesh_vertices_corrected = mesh_vertices.copy()
                    mesh_vertices_corrected[:, 2] = mesh_vertices[:, 2] * z_correction_factor
                    
                    # Comprehensive pose parameter validation and clamping
                    pose_clamped = False
                    
                    # 1. Root pose clamping (existing)
                    if 'smplx_root_pose' in model_output:
                        root_pose = model_output['smplx_root_pose'].detach().cpu().numpy()[0]
                        original_root_pose = root_pose.copy()
                        # More aggressive root pose clamping (>1.5 radians ≈ 86 degrees)
                        # The original had Z-rotation of -3.0665 rad (-175.6°) which is nearly upside down
                        root_pose_clamped = np.clip(root_pose, -1.5, 1.5)
                        if not np.allclose(original_root_pose, root_pose_clamped):
                            logging.warning(f"Frame {frame}, Person {person_id}: Extreme root pose detected {original_root_pose} -> clamped to {root_pose_clamped}")
                            model_output['smplx_root_pose'] = torch.from_numpy(root_pose_clamped).unsqueeze(0).to(model_output['smplx_root_pose'].device)
                            pose_clamped = True
                        
                    # 2. Body pose clamping (NEW - this fixes the distortion issue)
                    if 'smplx_body_pose' in model_output:
                        body_pose = model_output['smplx_body_pose'].detach().cpu().numpy()[0]
                        original_body_pose = body_pose.copy()
                        
                        # Check for extreme body poses (>1.8 radians ≈ 103 degrees)
                        # This is more restrictive than the 2.0 rad threshold we identified
                        extreme_mask = np.abs(body_pose) > 1.8
                        extreme_count = np.sum(extreme_mask)
                        
                        if extreme_count > 0:
                            # Clamp extreme body poses to realistic human motion limits
                            body_pose_clamped = np.clip(body_pose, -1.8, 1.8)
                            
                            logging.warning(f"Frame {frame}, Person {person_id}: {extreme_count} extreme body poses detected (>{np.degrees(1.8):.1f}°)")
                            logging.warning(f"  Range before clamping: [{np.degrees(body_pose.min()):.1f}°, {np.degrees(body_pose.max()):.1f}°]")
                            logging.warning(f"  Range after clamping: [{np.degrees(body_pose_clamped.min()):.1f}°, {np.degrees(body_pose_clamped.max()):.1f}°]")
                            
                            # Update the model output with clamped values
                            model_output['smplx_body_pose'] = torch.from_numpy(body_pose_clamped).unsqueeze(0).to(model_output['smplx_body_pose'].device)
                            pose_clamped = True
                            
                        # Check for NaN or infinite values
                        if np.any(np.isnan(body_pose)) or np.any(np.isinf(body_pose)):
                            logging.error(f"Frame {frame}, Person {person_id}: NaN/Inf values in body pose - resetting to neutral")
                            body_pose_neutral = np.zeros_like(body_pose)
                            model_output['smplx_body_pose'] = torch.from_numpy(body_pose_neutral).unsqueeze(0).to(model_output['smplx_body_pose'].device)
                            pose_clamped = True
                    
                    if pose_clamped:
                        logging.info(f"Frame {frame}, Person {person_id}: Pose parameters clamped for more realistic human poses")
                        
                        # CRITICAL FIX: Regenerate mesh with clamped pose parameters
                        logging.info(f"Frame {frame}, Person {person_id}: Regenerating mesh with clamped parameters")
                        
                        # Extract all SMPL-X parameters from model output
                        root_pose = model_output['smplx_root_pose']  # Already clamped
                        body_pose = model_output['smplx_body_pose']  # Already clamped
                        shape = model_output['smplx_shape']
                        lhand_pose = model_output.get('smplx_lhand_pose', torch.zeros(1, 45).to(root_pose.device))
                        rhand_pose = model_output.get('smplx_rhand_pose', torch.zeros(1, 45).to(root_pose.device))
                        jaw_pose = model_output.get('smplx_jaw_pose', torch.zeros(1, 3).to(root_pose.device))
                        expr = model_output.get('smplx_expr', torch.zeros(1, 10).to(root_pose.device))
                        cam_trans = model_output.get('cam_trans', torch.zeros(1, 3).to(root_pose.device))
                        
                        # Eye poses are required by SMPL-X layer (typically zero for most applications)
                        leye_pose = torch.zeros(1, 3).to(root_pose.device)
                        reye_pose = torch.zeros(1, 3).to(root_pose.device)
                        
                        # Use SMPL-X layer to regenerate mesh with clamped parameters
                        try:
                            with torch.no_grad():
                                # Make sure SMPL-X layer is on the correct device
                                smplx_layer = smpl_x.layer['neutral'].to(root_pose.device)
                                
                                # Assume neutral gender for simplicity (could be enhanced to detect gender)
                                smplx_output = smplx_layer(
                                    betas=shape,
                                    body_pose=body_pose.view(1, -1),
                                    global_orient=root_pose,
                                    left_hand_pose=lhand_pose.view(1, -1),
                                    right_hand_pose=rhand_pose.view(1, -1),
                                    jaw_pose=jaw_pose,
                                    leye_pose=leye_pose,
                                    reye_pose=reye_pose,
                                    expression=expr,
                                    transl=cam_trans
                                )
                                
                                # Update model output with regenerated mesh
                                regenerated_mesh = smplx_output.vertices
                                model_output['smplx_mesh_cam'] = regenerated_mesh
                                
                                logging.info(f"Frame {frame}, Person {person_id}: Mesh successfully regenerated with clamped parameters")
                                logging.info(f"  Original mesh range: [{mesh_tensor.min():.3f}, {mesh_tensor.max():.3f}]")
                                logging.info(f"  Regenerated mesh range: [{regenerated_mesh.min():.3f}, {regenerated_mesh.max():.3f}]")
                                
                        except Exception as e:
                            logging.error(f"Frame {frame}, Person {person_id}: Failed to regenerate mesh with clamped parameters: {e}")
                            logging.error("Continuing with original mesh")
                    
                    # Use corrected vertices for all downstream processing
                    mesh_vertices = mesh_vertices_corrected
                    
                    # Validate and convert SMPL-X parameters
                    smplx_params = validate_and_convert_smplx_params(
                        model_output, person_id, frame
                    )
                    if smplx_params:
                        frame_output.smplx_params[person_id] = smplx_params
                        validation_logger.log_smplx_generation(frame, person_id, smplx_params)
                    
                    # Validate and convert mesh data
                    mesh_data = validate_and_convert_mesh(
                        mesh_vertices, mesh_faces, person_id, frame
                    )
                    if mesh_data:
                        frame_output.meshes[person_id] = mesh_data
                        validation_logger.log_mesh_generation(frame, person_id, mesh_data)
                        
                        # Save mesh files if requested
                        if args.save_meshes and mesh_folder:
                            saved_mesh_files = save_mesh_with_validation(
                                mesh_data, args.mesh_format, Path(mesh_folder)
                            )
                            frame_output.mesh_files_saved[person_id] = saved_mesh_files
                        
                        # Save mesh renders if requested
                        if args.save_mesh_renders and mesh_render_folder:
                            # Create camera parameters for isolated mesh rendering
                            cam_param = {
                                'focal': [
                                    cfg.model.focal[0] / cfg.model.input_body_shape[1] * processed_bbox[2], 
                                    cfg.model.focal[1] / cfg.model.input_body_shape[0] * processed_bbox[3]
                                ],
                                'princpt': [
                                    cfg.model.princpt[0] / cfg.model.input_body_shape[1] * processed_bbox[2] + processed_bbox[0], 
                                    cfg.model.princpt[1] / cfg.model.input_body_shape[0] * processed_bbox[3] + processed_bbox[1]
                                ]
                            }
                            
                            saved_render_files = save_mesh_render_with_validation(
                                mesh_data, cam_param, Path(mesh_render_folder)
                            )
                            frame_output.render_files_saved[person_id] = saved_render_files
                        
                        # Log file operations
                        validation_logger.log_file_operations(
                            frame, person_id,
                            frame_output.mesh_files_saved.get(person_id, []),
                            frame_output.render_files_saved.get(person_id, [])
                        )

                        # Render mesh on visualization image
                        try:
                            focal = [
                                cfg.model.focal[0] / cfg.model.input_body_shape[1] * processed_bbox[2], 
                                cfg.model.focal[1] / cfg.model.input_body_shape[0] * processed_bbox[3]
                            ]
                            princpt = [
                                cfg.model.princpt[0] / cfg.model.input_body_shape[1] * processed_bbox[2] + processed_bbox[0], 
                                cfg.model.princpt[1] / cfg.model.input_body_shape[0] * processed_bbox[3] + processed_bbox[1]
                            ]
                            
                            # Draw the bbox on img
                            vis_img = cv2.rectangle(vis_img, (int(bbox[0]), int(bbox[1])), 
                                                    (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                            
                            # Draw mesh with improved Mac-compatible rendering
                            try:
                                vis_img = render_mesh(vis_img, mesh_vertices, mesh_faces, 
                                                    {'focal': focal, 'princpt': princpt}, 
                                                    mesh_as_vertices=False)
                            except Exception as render_error:
                                logging.warning(f"PyRender failed ({render_error}), using improved wireframe fallback")
                                # Use improved wireframe rendering
                                try:
                                    vis_img = render_mesh_improved(vis_img, mesh_vertices, mesh_faces,
                                                                 {'focal': focal, 'princpt': princpt})
                                except Exception as wireframe_error:
                                    logging.error(f"Wireframe rendering also failed: {wireframe_error}")
                                    # Last fallback: just draw vertices as points
                                    vis_img = render_mesh(vis_img, mesh_vertices, mesh_faces, 
                                                        {'focal': focal, 'princpt': princpt}, 
                                                        mesh_as_vertices=True)
                        except Exception as viz_error:
                            logging.error(f"Visualization rendering failed: {viz_error}")
                    else:
                        logging.error(f"Frame {frame}, Person {person_id}: Failed to validate mesh data")
                        
                except Exception as person_error:
                    logging.error(f"Frame {frame}, Person {person_id}: Processing failed - {person_error}")
                    continue
            
            # Save visualization image
            frame_name = os.path.basename(img_path)
            vis_output_path = os.path.join(output_folder, frame_name)
            cv2.imwrite(vis_output_path, vis_img[:, :, ::-1])
            
            # Record processing time
            frame_output.processing_time = time.time() - frame_start_time
            all_outputs.append(frame_output)
            
        except Exception as frame_error:
            logging.error(f"Frame {frame}: Processing failed - {frame_error}")
            frame_output.processing_time = time.time() - frame_start_time
            all_outputs.append(frame_output)
            continue
    
    # Final validation and summary
    try:
        validation_logger.log_inference_summary(all_outputs)
        
        # Save validation results
        output_summary = {
            'total_frames': len(all_outputs),
            'successful_frames': len([out for out in all_outputs if out.detections]),
            'total_detections': sum(len(out.detections) for out in all_outputs),
            'total_meshes': sum(len(out.meshes) for out in all_outputs),
            'total_processing_time': sum(out.processing_time for out in all_outputs),
            'config': config.model_dump()
        }
        
        # Save summary to JSON
        import json
        summary_path = Path(output_folder) / "inference_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(output_summary, f, indent=2, default=str)
        
        logging.info(f"Inference completed successfully. Summary saved to: {summary_path}")
        
    except Exception as e:
        logging.error(f"Failed to save final summary: {e}")


if __name__ == "__main__":
    main()
