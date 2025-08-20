"""
Pydantic models for SMPLest-X inference validation and logging.
Ensures proper data flow and validation at every step of the inference pipeline.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional, Dict, Tuple, Annotated
from pathlib import Path
import numpy as np
import torch
from enum import Enum
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('inference_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('SMPLestX_Validation')

class DeviceType(str, Enum):
    """Supported device types"""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"

class MeshFormat(str, Enum):
    """Supported mesh formats"""
    OBJ = "obj"
    PLY = "ply"

class InferenceConfig(BaseModel):
    """Configuration validation for inference parameters"""
    
    # Input/Output paths
    input_path: Path = Field(..., description="Path to input video or image")
    output_folder: Path = Field(..., description="Output folder path")
    model_path: Path = Field(..., description="Path to model checkpoint")
    
    # Processing parameters
    gpu: int = Field(0, ge=0, description="GPU ID to use")
    bbox_thr: float = Field(0.9, ge=0.0, le=1.0, description="Bounding box threshold")
    fps: Optional[int] = Field(None, ge=1, le=120, description="Frame rate for video processing")
    
    # Mesh saving options
    save_meshes: bool = Field(False, description="Save 3D meshes")
    save_mesh_renders: bool = Field(False, description="Save mesh render images")
    mesh_format: MeshFormat = Field(MeshFormat.OBJ, description="Mesh file format")
    
    # Device configuration
    device: Optional[DeviceType] = Field(None, description="Computation device")
    
    @field_validator('input_path')
    @classmethod
    def validate_input_exists(cls, v):
        if not v.exists():
            raise ValueError(f"Input file does not exist: {v}")
        return v
    
    @field_validator('model_path')
    @classmethod
    def validate_model_exists(cls, v):
        if not v.exists():
            raise ValueError(f"Model file does not exist: {v}")
        return v
    
    @field_validator('output_folder')
    @classmethod
    def create_output_folder(cls, v):
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    class Config:
        use_enum_values = True

class DetectionResult(BaseModel):
    """Validation for person detection results"""
    
    person_id: int = Field(..., ge=0, description="Person ID in the frame")
    bbox: Annotated[List[float], Field(min_length=4, max_length=4)] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    frame_id: int = Field(..., ge=0, description="Frame number")
    
    @field_validator('bbox')
    @classmethod
    def validate_bbox_format(cls, v):
        x1, y1, x2, y2 = v
        if x1 >= x2 or y1 >= y2:
            raise ValueError(f"Invalid bounding box format: {v}")
        if any(coord < 0 for coord in v):
            raise ValueError(f"Negative coordinates in bbox: {v}")
        return v

class SMPLXParameters(BaseModel):
    """Validation for SMPL-X model parameters"""
    
    betas: List[float] = Field(..., description="Shape parameters")
    body_pose: List[float] = Field(..., description="Body pose parameters") 
    global_orient: List[float] = Field(..., description="Global orientation")
    left_hand_pose: Optional[List[float]] = Field(None, description="Left hand pose")
    right_hand_pose: Optional[List[float]] = Field(None, description="Right hand pose")
    jaw_pose: Optional[List[float]] = Field(None, description="Jaw pose")
    leye_pose: Optional[List[float]] = Field(None, description="Left eye pose")
    reye_pose: Optional[List[float]] = Field(None, description="Right eye pose")
    expression: Optional[List[float]] = Field(None, description="Expression parameters")
    transl: Annotated[List[float], Field(min_length=3, max_length=3)] = Field(..., description="Translation")
    
    @field_validator('betas')
    @classmethod
    def validate_betas_length(cls, v):
        if len(v) not in [10, 300]:  # Common SMPL-X shape parameter counts
            logger.warning(f"Unusual number of shape parameters: {len(v)}")
        return v
    
    @field_validator('body_pose')
    @classmethod
    def validate_body_pose_length(cls, v):
        expected_length = 63  # 21 joints * 3 parameters
        if len(v) != expected_length:
            logger.warning(f"Body pose length {len(v)} != expected {expected_length}")
        return v

class MeshData(BaseModel):
    """Validation for generated mesh data"""
    
    vertices: List[List[float]] = Field(..., description="Mesh vertices")
    faces: List[List[int]] = Field(..., description="Mesh faces")
    person_id: int = Field(..., ge=0, description="Person ID")
    frame_id: int = Field(..., ge=0, description="Frame ID")
    
    @field_validator('vertices')
    @classmethod
    def validate_vertices_format(cls, v):
        if not v:
            raise ValueError("Empty vertices array")
        
        for i, vertex in enumerate(v):
            if len(vertex) != 3:
                raise ValueError(f"Vertex {i} must have 3 coordinates, got {len(vertex)}")
            if any(not isinstance(coord, (int, float)) for coord in vertex):
                raise ValueError(f"Invalid vertex coordinates at {i}: {vertex}")
        
        logger.info(f"Validated {len(v)} vertices")
        return v
    
    @field_validator('faces')
    @classmethod
    def validate_faces_format(cls, v):
        if not v:
            raise ValueError("Empty faces array")
        
        for i, face in enumerate(v):
            if len(face) != 3:
                raise ValueError(f"Face {i} must have 3 vertex indices, got {len(face)}")
            if any(not isinstance(idx, int) or idx < 0 for idx in face):
                raise ValueError(f"Invalid face indices at {i}: {face}")
        
        logger.info(f"Validated {len(v)} faces")
        return v
    
    @model_validator(mode='after')
    def validate_faces_reference_vertices(self):
        vertices = self.vertices
        faces = self.faces
        
        if vertices and faces:
            max_vertex_idx = len(vertices) - 1
            for i, face in enumerate(faces):
                for vertex_idx in face:
                    if vertex_idx > max_vertex_idx:
                        raise ValueError(f"Face {i} references vertex {vertex_idx} > max {max_vertex_idx}")
        
        return self

class InferenceOutput(BaseModel):
    """Validation for complete inference output"""
    
    frame_id: int = Field(..., ge=0, description="Frame number")
    detections: List[DetectionResult] = Field(default_factory=list, description="Person detections")
    smplx_params: Dict[int, SMPLXParameters] = Field(default_factory=dict, description="SMPL-X parameters by person ID")
    meshes: Dict[int, MeshData] = Field(default_factory=dict, description="Generated meshes by person ID")
    mesh_files_saved: Dict[int, List[Path]] = Field(default_factory=dict, description="Saved mesh file paths")
    render_files_saved: Dict[int, List[Path]] = Field(default_factory=dict, description="Saved render file paths")
    processing_time: float = Field(..., ge=0.0, description="Processing time in seconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Processing timestamp")
    
    @field_validator('processing_time')
    @classmethod
    def validate_reasonable_processing_time(cls, v):
        if v > 300:  # 5 minutes per frame seems excessive
            logger.warning(f"Very long processing time: {v:.2f}s")
        return v
    
    @model_validator(mode='after')
    def validate_consistency(self):
        detections = self.detections
        smplx_params = self.smplx_params
        meshes = self.meshes
        
        # Check that we have SMPL-X parameters for each detection
        detection_ids = {det.person_id for det in detections}
        param_ids = set(smplx_params.keys())
        mesh_ids = set(meshes.keys())
        
        if detection_ids and not param_ids:
            logger.error(f"No SMPL-X parameters generated for detections: {detection_ids}")
        
        missing_params = detection_ids - param_ids
        if missing_params:
            logger.warning(f"Missing SMPL-X parameters for persons: {missing_params}")
        
        missing_meshes = detection_ids - mesh_ids
        if missing_meshes:
            logger.warning(f"Missing meshes for persons: {missing_meshes}")
        
        return self

class ValidationLogger:
    """Enhanced logging for validation and debugging"""
    
    def __init__(self, log_file: Optional[Path] = None):
        self.logger = logging.getLogger('SMPLestX_Validation')
        if log_file:
            handler = logging.FileHandler(log_file)
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            self.logger.addHandler(handler)
    
    def log_config_validation(self, config: InferenceConfig):
        """Log configuration validation results"""
        self.logger.info("=" * 50)
        self.logger.info("INFERENCE CONFIGURATION VALIDATION")
        self.logger.info("=" * 50)
        self.logger.info(f"Input: {config.input_path}")
        self.logger.info(f"Output: {config.output_folder}")
        self.logger.info(f"Model: {config.model_path}")
        self.logger.info(f"Device: {config.device}")
        self.logger.info(f"Bbox threshold: {config.bbox_thr}")
        self.logger.info(f"Save meshes: {config.save_meshes}")
        self.logger.info(f"Save renders: {config.save_mesh_renders}")
        self.logger.info(f"Mesh format: {config.mesh_format}")
    
    def log_frame_processing(self, frame_id: int, detections: List[DetectionResult]):
        """Log frame processing results"""
        self.logger.info(f"Frame {frame_id:06d}: {len(detections)} detections")
        for det in detections:
            self.logger.info(f"  Person {det.person_id}: bbox={det.bbox}, conf={det.confidence:.3f}")
    
    def log_smplx_generation(self, frame_id: int, person_id: int, params: SMPLXParameters):
        """Log SMPL-X parameter generation"""
        self.logger.info(f"Frame {frame_id:06d}, Person {person_id}: SMPL-X generated")
        self.logger.info(f"  Shape params: {len(params.betas)}")
        self.logger.info(f"  Body pose: {len(params.body_pose)}")
        self.logger.info(f"  Translation: {params.transl}")
    
    def log_mesh_generation(self, frame_id: int, person_id: int, mesh: MeshData):
        """Log mesh generation results"""
        self.logger.info(f"Frame {frame_id:06d}, Person {person_id}: Mesh generated")
        self.logger.info(f"  Vertices: {len(mesh.vertices)}")
        self.logger.info(f"  Faces: {len(mesh.faces)}")
    
    def log_file_operations(self, frame_id: int, person_id: int, mesh_files: List[Path], render_files: List[Path]):
        """Log file save operations"""
        self.logger.info(f"Frame {frame_id:06d}, Person {person_id}: Files saved")
        for file_path in mesh_files:
            self.logger.info(f"  Mesh: {file_path}")
        for file_path in render_files:
            self.logger.info(f"  Render: {file_path}")
    
    def log_inference_summary(self, outputs: List[InferenceOutput]):
        """Log complete inference summary"""
        self.logger.info("=" * 50)
        self.logger.info("INFERENCE SUMMARY")
        self.logger.info("=" * 50)
        
        total_frames = len(outputs)
        total_detections = sum(len(out.detections) for out in outputs)
        total_meshes = sum(len(out.meshes) for out in outputs)
        total_time = sum(out.processing_time for out in outputs)
        
        self.logger.info(f"Total frames processed: {total_frames}")
        self.logger.info(f"Total detections: {total_detections}")
        self.logger.info(f"Total meshes generated: {total_meshes}")
        self.logger.info(f"Total processing time: {total_time:.2f}s")
        self.logger.info(f"Average time per frame: {total_time/total_frames:.2f}s" if total_frames > 0 else "No frames processed")
        
        # Check for frames with no output
        empty_frames = [out.frame_id for out in outputs if not out.detections]
        if empty_frames:
            self.logger.warning(f"Frames with no detections: {len(empty_frames)} ({empty_frames[:10]}{'...' if len(empty_frames) > 10 else ''})")
        
        # Check for failed mesh generations
        failed_mesh_frames = [out.frame_id for out in outputs if out.detections and not out.meshes]
        if failed_mesh_frames:
            self.logger.error(f"Frames with detections but no meshes: {len(failed_mesh_frames)} ({failed_mesh_frames[:10]}{'...' if len(failed_mesh_frames) > 10 else ''})")

def validate_tensor_output(tensor: torch.Tensor, name: str, expected_shape: Optional[Tuple] = None) -> bool:
    """Validate tensor outputs from model"""
    try:
        if not isinstance(tensor, torch.Tensor):
            logger.error(f"{name}: Expected torch.Tensor, got {type(tensor)}")
            return False
        
        if torch.isnan(tensor).any():
            logger.error(f"{name}: Contains NaN values")
            return False
        
        if torch.isinf(tensor).any():
            logger.error(f"{name}: Contains infinite values")
            return False
        
        if expected_shape and tensor.shape != expected_shape:
            logger.warning(f"{name}: Shape {tensor.shape} != expected {expected_shape}")
        
        logger.info(f"{name}: Valid tensor {tensor.shape}, range [{tensor.min():.3f}, {tensor.max():.3f}]")
        return True
        
    except Exception as e:
        logger.error(f"{name}: Validation failed - {e}")
        return False

def validate_numpy_array(array: np.ndarray, name: str, expected_shape: Optional[Tuple] = None) -> bool:
    """Validate numpy array outputs"""
    try:
        if not isinstance(array, np.ndarray):
            logger.error(f"{name}: Expected numpy.ndarray, got {type(array)}")
            return False
        
        if np.isnan(array).any():
            logger.error(f"{name}: Contains NaN values")
            return False
        
        if np.isinf(array).any():
            logger.error(f"{name}: Contains infinite values")
            return False
        
        if expected_shape and array.shape != expected_shape:
            logger.warning(f"{name}: Shape {array.shape} != expected {expected_shape}")
        
        logger.info(f"{name}: Valid array {array.shape}, range [{array.min():.3f}, {array.max():.3f}]")
        return True
        
    except Exception as e:
        logger.error(f"{name}: Validation failed - {e}")
        return False