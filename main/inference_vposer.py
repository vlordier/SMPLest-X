"""
Enhanced inference script with VPoser integration and PyTorch3D rendering
Supports both traditional pose estimation and VPoser-regularized inference
"""

import os
import sys
import cv2
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main.config import cfg
from models.SMPLest_X_VPoser import get_model_with_vposer
from human_models.human_models import SMPLX
from utils.inference_utils import InferenceRunner
from utils.vposer_utils import create_vposer_wrapper
from utils.pytorch3d_renderer import create_pytorch3d_renderer
from utils.pyvista_renderer import PyVistaRenderer


class VPoserInferenceRunner(InferenceRunner):
    """
    Enhanced inference runner with VPoser and PyTorch3D support
    """
    
    def __init__(self, config_path, checkpoint_path, use_vposer=True, use_pytorch3d=True):
        """
        Initialize enhanced inference runner
        
        Args:
            config_path: Path to config file
            checkpoint_path: Path to model checkpoint
            use_vposer: Enable VPoser pose regularization
            use_pytorch3d: Use PyTorch3D renderer (fallback to PyVista if unavailable)
        """
        # Load configuration
        cfg.merge_from_file(config_path)
        cfg.freeze()
        
        self.cfg = cfg
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_vposer = use_vposer
        self.use_pytorch3d = use_pytorch3d
        
        # Initialize model
        self._setup_model(checkpoint_path)
        
        # Initialize renderers
        self._setup_renderers()
        
        # Initialize VPoser if enabled
        self.vposer_wrapper = None
        if use_vposer:
            self._setup_vposer()
        
        logging.info(f"VPoser inference runner initialized")
        logging.info(f"VPoser enabled: {self.use_vposer and self.vposer_wrapper is not None}")
        logging.info(f"PyTorch3D renderer: {self.pytorch3d_renderer is not None}")
    
    def _setup_model(self, checkpoint_path):
        """Setup model with VPoser support"""
        # Modify config for VPoser if enabled
        if self.use_vposer:
            cfg.model.use_vposer = True
            cfg.model.vposer_ckpt_dir = './data/vposer_v1_0/snapshots'  # Default path
            
        # Create model
        self.model = get_model_with_vposer(cfg, 'test')
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'network' in checkpoint:
            state_dict = checkpoint['network']
        else:
            state_dict = checkpoint
            
        # Load state dict
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            logging.warning(f"Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            logging.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")
            
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logging.info(f"Model loaded from {checkpoint_path}")
    
    def _setup_renderers(self):
        """Setup rendering components"""
        # Try PyTorch3D first
        if self.use_pytorch3d:
            self.pytorch3d_renderer = create_pytorch3d_renderer(
                image_size=(512, 512),
                device=self.device
            )
        else:
            self.pytorch3d_renderer = None
            
        # Fallback to PyVista
        try:
            self.pyvista_renderer = PyVistaRenderer()
        except Exception as e:
            logging.warning(f"PyVista renderer initialization failed: {e}")
            self.pyvista_renderer = None
        
        # SMPL-X instance for mesh faces
        self.smplx = SMPLX.get_instance()
    
    def _setup_vposer(self):
        """Setup VPoser wrapper"""
        try:
            vposer_ckpt_dir = getattr(self.cfg.model, 'vposer_ckpt_dir', './data/vposer_v1_0/snapshots')
            self.vposer_wrapper = create_vposer_wrapper(vposer_ckpt_dir, self.device)
            
            if self.vposer_wrapper:
                logging.info("VPoser wrapper initialized successfully")
            else:
                logging.warning("VPoser initialization failed")
                
        except Exception as e:
            logging.warning(f"VPoser setup error: {e}")
            self.vposer_wrapper = None
    
    def process_pose_with_vposer(self, body_pose, regularization_strength=0.3):
        """
        Apply VPoser regularization to body pose
        
        Args:
            body_pose: Raw body pose predictions [63]
            regularization_strength: VPoser regularization strength
            
        Returns:
            regularized_pose: VPoser-regularized body pose
        """
        if not self.vposer_wrapper:
            return body_pose
            
        try:
            # Ensure tensor format
            if isinstance(body_pose, np.ndarray):
                body_pose = torch.from_numpy(body_pose).float().to(self.device)
            
            # Add batch dimension if needed
            if body_pose.dim() == 1:
                body_pose = body_pose.unsqueeze(0)
                squeeze_output = True
            else:
                squeeze_output = False
            
            # Apply VPoser regularization
            with torch.no_grad():
                regularized_pose = self.vposer_wrapper.regularize_pose(
                    body_pose, alpha=regularization_strength
                )
            
            # Remove batch dimension if added
            if squeeze_output:
                regularized_pose = regularized_pose.squeeze(0)
                
            return regularized_pose
            
        except Exception as e:
            logging.warning(f"VPoser regularization failed: {e}")
            return body_pose
    
    def render_mesh_enhanced(self, vertices, person_id, frame_idx, 
                           overlay_image=None, save_path=None):
        """
        Enhanced mesh rendering with multiple renderer options
        
        Args:
            vertices: Mesh vertices
            person_id: Person ID
            frame_idx: Frame index
            overlay_image: Optional background image
            save_path: Path to save rendered image
            
        Returns:
            rendered_image: Final rendered image
        """
        try:
            # Try PyTorch3D first
            if self.pytorch3d_renderer:
                rendered_image = self._render_with_pytorch3d(
                    vertices, overlay_image
                )
                logging.debug(f"Frame {frame_idx}, Person {person_id}: Rendered with PyTorch3D")
            
            # Fallback to PyVista
            elif self.pyvista_renderer:
                rendered_image = self._render_with_pyvista(
                    vertices, overlay_image
                )
                logging.debug(f"Frame {frame_idx}, Person {person_id}: Rendered with PyVista")
            
            else:
                # Create simple wireframe if no renderer available
                rendered_image = self._create_simple_visualization(
                    vertices, overlay_image
                )
                logging.debug(f"Frame {frame_idx}, Person {person_id}: Simple visualization")
            
            # Save rendered image if path provided
            if save_path:
                cv2.imwrite(save_path, cv2.cvtColor(rendered_image, cv2.COLOR_RGB2BGR))
                
            return rendered_image
            
        except Exception as e:
            logging.error(f"Frame {frame_idx}, Person {person_id}: Rendering failed: {e}")
            return overlay_image if overlay_image is not None else np.zeros((512, 512, 3), dtype=np.uint8)
    
    def _render_with_pytorch3d(self, vertices, overlay_image=None):
        """Render mesh using PyTorch3D"""
        faces = torch.from_numpy(self.smplx.face.astype(np.int64))
        
        rendered_image = self.pytorch3d_renderer.render_smplx_mesh(
            smplx_vertices=vertices,
            smplx_faces=faces,
            camera_params={'distance': 2.5, 'elevation': 0, 'azimuth': 0},
            overlay_image=overlay_image,
            alpha=0.8
        )
        
        return rendered_image
    
    def _render_with_pyvista(self, vertices, overlay_image=None):
        """Render mesh using PyVista"""
        mesh_image = self.pyvista_renderer.render_mesh(
            vertices.cpu().numpy() if isinstance(vertices, torch.Tensor) else vertices,
            self.smplx.face
        )
        
        if overlay_image is not None:
            # Simple alpha blending
            alpha = 0.8
            mask = np.any(mesh_image > 10, axis=2)
            result = overlay_image.copy()
            result[mask] = (alpha * mesh_image[mask] + 
                          (1 - alpha) * overlay_image[mask]).astype(np.uint8)
            return result
        
        return mesh_image
    
    def _create_simple_visualization(self, vertices, overlay_image=None):
        """Create simple point visualization when no renderer available"""
        if overlay_image is None:
            base_image = np.zeros((512, 512, 3), dtype=np.uint8)
        else:
            base_image = overlay_image.copy()
        
        # Project 3D points to 2D for simple visualization
        if isinstance(vertices, torch.Tensor):
            vertices = vertices.cpu().numpy()
        
        # Simple orthographic projection
        x = ((vertices[:, 0] + 1) * 256).astype(np.int32)
        y = ((vertices[:, 1] + 1) * 256).astype(np.int32)
        
        # Draw points
        valid_mask = (x >= 0) & (x < 512) & (y >= 0) & (y < 512)
        for i in range(len(x)):
            if valid_mask[i]:
                cv2.circle(base_image, (x[i], y[i]), 1, (0, 255, 0), -1)
        
        return base_image
    
    def run_enhanced_inference(self, video_path, output_dir, 
                             vposer_regularization=0.3,
                             save_mesh_renders=True,
                             frame_skip=2):
        """
        Run enhanced inference with VPoser and improved rendering
        
        Args:
            video_path: Input video file path
            output_dir: Output directory
            vposer_regularization: VPoser regularization strength (0.0-1.0)
            save_mesh_renders: Whether to save mesh renders
            frame_skip: Process every N-th frame
        """
        # Setup output directories
        base_name = Path(video_path).stem
        frame_output_dir = Path(output_dir) / 'output_frames' / base_name
        mesh_output_dir = Path(output_dir) / 'mesh_renders' / base_name
        
        frame_output_dir.mkdir(parents=True, exist_ok=True)
        if save_mesh_renders:
            mesh_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup video capture
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        logging.info(f"Processing video: {video_path}")
        logging.info(f"Total frames: {total_frames}, FPS: {fps}")
        logging.info(f"Frame skip: {frame_skip} (processing every {frame_skip} frames)")
        logging.info(f"VPoser regularization: {vposer_regularization}")
        
        frame_idx = 0
        processed_frames = 0
        
        # Process video frames
        pbar = tqdm(total=total_frames // frame_skip, desc="Processing frames")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames based on frame_skip
                if frame_idx % frame_skip != 0:
                    frame_idx += 1
                    continue
                
                try:
                    # Run inference on frame
                    results = self.process_frame_enhanced(
                        frame, frame_idx, vposer_regularization
                    )
                    
                    if results:
                        # Save output frame
                        output_frame_path = frame_output_dir / f"{frame_idx:06d}.jpg"
                        cv2.imwrite(str(output_frame_path), frame)
                        
                        # Process each detected person
                        for person_id, result in enumerate(results):
                            if save_mesh_renders and 'smplx_mesh_cam' in result:
                                # Render and save mesh
                                mesh_path = mesh_output_dir / f"{frame_idx:06d}_person{person_id}_mesh.jpg"
                                self.render_mesh_enhanced(
                                    result['smplx_mesh_cam'],
                                    person_id,
                                    frame_idx,
                                    overlay_image=frame,
                                    save_path=str(mesh_path)
                                )
                    
                    processed_frames += 1
                    
                except Exception as e:
                    logging.error(f"Frame {frame_idx} processing failed: {e}")
                
                frame_idx += 1
                pbar.update(1)
                
        finally:
            cap.release()
            pbar.close()
        
        logging.info(f"Processed {processed_frames} frames")
        logging.info(f"Output frames saved to: {frame_output_dir}")
        if save_mesh_renders:
            logging.info(f"Mesh renders saved to: {mesh_output_dir}")
    
    def process_frame_enhanced(self, frame, frame_idx, vposer_regularization=0.3):
        """
        Process single frame with VPoser enhancement
        
        Args:
            frame: Input frame image
            frame_idx: Frame index
            vposer_regularization: VPoser regularization strength
            
        Returns:
            results: List of detection/estimation results
        """
        import torch
        import numpy as np
        from utils.human_models import smpl_x
        from utils.preprocessing import process_bbox, generate_patch_image, non_max_suppression
        from torchvision import transforms as T
        
        results = []
        
        try:
            # Person detection using YOLO
            yolo_bbox = self.yolo_model(frame)
            
            if len(yolo_bbox) == 0:
                logging.debug(f"Frame {frame_idx}: No persons detected")
                return results
            
            # Apply NMS if multi-person
            if self.multi_person:
                yolo_bbox = non_max_suppression(yolo_bbox, self.iou_threshold)
            else:
                yolo_bbox = yolo_bbox[:1]  # Only largest bbox
            
            original_img_height, original_img_width = frame.shape[:2]
            
            # Process each detected person
            for bbox_id, bbox_xyxy in enumerate(yolo_bbox):
                # Convert YOLO bbox to xywh format
                yolo_bbox_xywh = np.array([
                    bbox_xyxy[0],  # x
                    bbox_xyxy[1],  # y 
                    abs(bbox_xyxy[2] - bbox_xyxy[0]),  # width
                    abs(bbox_xyxy[3] - bbox_xyxy[1])   # height
                ])
                
                # Process bbox for model input
                bbox = process_bbox(
                    bbox=yolo_bbox_xywh,
                    img_width=original_img_width,
                    img_height=original_img_height,
                    input_img_shape=self.input_img_shape,
                    ratio=1.25
                )
                
                # Generate patch image
                img, _, _ = generate_patch_image(
                    cvimg=frame,
                    bbox=bbox,
                    scale=1.0,
                    rot=0.0,
                    do_flip=False,
                    out_shape=self.input_img_shape
                )
                
                # Transform and normalize
                transform = T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
                
                img_tensor = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1) / 255.0
                img_tensor = transform(img_tensor).unsqueeze(0)
                
                # Move to device
                device = next(self.model.parameters()).device
                img_tensor = img_tensor.to(device)
                
                # Model inference
                inputs = {'img': img_tensor}
                targets = {}
                meta_info = {}
                
                with torch.no_grad():
                    out = self.model(inputs, targets, meta_info, 'test')
                
                # Apply VPoser regularization if available and enabled
                if self.vposer_model is not None and vposer_regularization > 0:
                    try:
                        # Get body pose from output
                        if 'body_pose' in out:
                            original_pose = out['body_pose']
                            
                            # Apply VPoser regularization
                            regularized_pose = self.apply_vposer_regularization(
                                original_pose, vposer_regularization
                            )
                            
                            # Update the output with regularized pose
                            out['body_pose'] = regularized_pose
                            
                            # Re-generate mesh with regularized pose if needed
                            if hasattr(self, 'regenerate_mesh_from_pose'):
                                out['smplx_mesh_cam'] = self.regenerate_mesh_from_pose(out)
                    
                    except Exception as e:
                        logging.warning(f"VPoser regularization failed for person {bbox_id}: {e}")
                
                # Extract mesh
                if 'smplx_mesh_cam' in out:
                    mesh = out['smplx_mesh_cam'].detach().cpu().numpy()[0]
                    
                    # Calculate camera parameters for rendering
                    focal = [
                        self.focal[0] / self.input_body_shape[1] * bbox[2],
                        self.focal[1] / self.input_body_shape[0] * bbox[3]
                    ]
                    princpt = [
                        self.princpt[0] / self.input_body_shape[1] * bbox[2] + bbox[0],
                        self.princpt[1] / self.input_body_shape[0] * bbox[3] + bbox[1]
                    ]
                    
                    # Store results
                    result = {
                        'person_id': bbox_id,
                        'bbox': bbox,
                        'smplx_mesh_cam': mesh,
                        'focal': focal,
                        'princpt': princpt,
                        'frame_idx': frame_idx,
                        'vposer_applied': self.vposer_model is not None and vposer_regularization > 0
                    }
                    
                    # Add additional outputs if available
                    for key in ['body_pose', 'global_orient', 'betas', 'expression', 'left_hand_pose', 'right_hand_pose']:
                        if key in out:
                            result[key] = out[key].detach().cpu().numpy()
                    
                    results.append(result)
                    logging.debug(f"Frame {frame_idx}: Processed person {bbox_id}")
                
        except Exception as e:
            logging.error(f"Frame {frame_idx} processing failed: {e}")
        
        return results


def main():
    """Main inference script with VPoser support"""
    parser = argparse.ArgumentParser(description="SMPLest-X inference with VPoser")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--video', type=str, required=True, help='Input video path')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--use_vposer', action='store_true', help='Enable VPoser regularization')
    parser.add_argument('--use_pytorch3d', action='store_true', help='Use PyTorch3D renderer')
    parser.add_argument('--vposer_regularization', type=float, default=0.3,
                       help='VPoser regularization strength (0.0-1.0)')
    parser.add_argument('--frame_skip', type=int, default=2, help='Process every N-th frame')
    parser.add_argument('--save_mesh_renders', action='store_true', help='Save mesh render images')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create inference runner
    runner = VPoserInferenceRunner(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        use_vposer=args.use_vposer,
        use_pytorch3d=args.use_pytorch3d
    )
    
    # Run inference
    runner.run_enhanced_inference(
        video_path=args.video,
        output_dir=args.output_dir,
        vposer_regularization=args.vposer_regularization,
        save_mesh_renders=args.save_mesh_renders,
        frame_skip=args.frame_skip
    )


if __name__ == "__main__":
    main()