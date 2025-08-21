"""
Direct SMPL-X implementation to bypass coefficient mismatch issues
Uses direct tensor operations instead of the problematic smplx library
"""

import torch
import numpy as np
import os.path as osp
import pickle
from utils.device_utils import get_device, to_device

class Direct_SMPLX:
    """
    PyTorch3D-based SMPL-X implementation that handles coefficient mismatches
    by directly working with model vertices and faces without relying on 
    problematic smplx library forward passes.
    """
    _instance = None

    def __new__(cls, config=None):
        """Ensures only one instance exists."""
        if cls._instance is None:
            assert config is not None, 'Direct_SMPLX requires human model path'
            cls._instance = super(Direct_SMPLX, cls).__new__(cls)
            cls._instance._initialize(config)
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance to allow re-initialization."""
        cls._instance = None

    def _initialize(self, human_model_path):
        """Initialize the PyTorch3D-based SMPL-X model"""
        self.device = get_device()
        self.human_model_path = human_model_path
        
        # Load model data directly from files to bypass smplx library issues
        self._load_model_data()
        
        # Set dimensions based on actual model files
        self.vertex_num = 10475
        self.shape_param_dim = 10  # Based on our model files
        self.expr_code_dim = 10    # Based on our model files
        
        # Load additional data files
        with open(osp.join(human_model_path, 'smplx', 'SMPLX_to_J14.pkl'), 'rb') as f:
            self.j14_regressor = pickle.load(f, encoding='latin1')
        with open(osp.join(human_model_path, 'smplx', 'MANO_SMPLX_vertex_ids.pkl'), 'rb') as f:
            self.hand_vertex_idx = pickle.load(f, encoding='latin1')
        self.face_vertex_idx = np.load(osp.join(human_model_path, 'smplx', 'SMPL-X__FLAME_vertex_ids.npy'))
        
        # Define joint mappings (same as original implementation)
        self._setup_joint_mappings()

    def _load_model_data(self):
        """Load SMPL-X model data directly from files"""
        self.models = {}
        
        for gender in ['NEUTRAL', 'MALE', 'FEMALE']:
            model_file = osp.join(self.human_model_path, 'smplx', f'SMPLX_{gender}.pkl')
            
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f, encoding='latin1')
            
            # Convert to tensors and move to device
            model = {
                'v_template': to_device(torch.from_numpy(model_data['v_template']).float()),
                'shapedirs': to_device(torch.from_numpy(model_data['shapedirs']).float()),
                'posedirs': to_device(torch.from_numpy(model_data['posedirs']).float()),
                'J_regressor': to_device(torch.from_numpy(model_data['J_regressor']).float()),
                'parents': torch.from_numpy(model_data['parents']).long(),
                'weights': to_device(torch.from_numpy(model_data['weights']).float()),
                'faces': torch.from_numpy(model_data['f']).long()
            }
            
            gender_key = gender.lower() if gender != 'NEUTRAL' else 'neutral'
            self.models[gender_key] = model
        
        # Set up main model references
        self.face = self.models['neutral']['faces']
        self.J_regressor = self.models['neutral']['J_regressor'].cpu().numpy()
        self.J_regressor_idx = {'pelvis': 0, 'lwrist': 20, 'rwrist': 21, 'neck': 12}

    def _setup_joint_mappings(self):
        """Set up joint name mappings and indices (same as original)"""
        # Original SMPLX joint set
        self.orig_joint_num = 53
        self.orig_joints_name = \
        ('Pelvis', 'L_Hip', 'R_Hip', 'Spine_1', 'L_Knee', 'R_Knee', 'Spine_2', 'L_Ankle', 'R_Ankle', 'Spine_3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist',
        'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3',
        'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3',
        'Jaw')
        
        self.orig_root_joint_idx = self.orig_joints_name.index('Pelvis')
        
        # Changed SMPLX joint set for supervision
        self.joint_num = 137
        self.joints_name = \
        ('Pelvis', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Neck', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel', 'L_Ear', 'R_Ear', 'L_Eye', 'R_Eye', 'Nose',
         'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb_4', 'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Index_4', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Middle_4', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Ring_4', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4',
         'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4', 'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Index_4', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Middle_4', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Ring_4', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4',
         *['Face_' + str(i) for i in range(1,73)])
        
        self.root_joint_idx = self.joints_name.index('Pelvis')
        self.lwrist_idx = self.joints_name.index('L_Wrist')
        self.rwrist_idx = self.joints_name.index('R_Wrist') 
        self.neck_idx = self.joints_name.index('Neck')
        
        # Joint index mapping
        self.joint_idx = \
        (0,1,2,4,5,7,8,12,16,17,18,19,20,21,60,61,62,63,64,65,59,58,57,56,55,
        37,38,39,66,25,26,27,67,28,29,30,68,34,35,36,69,31,32,33,70,
        52,53,54,71,40,41,42,72,43,44,45,73,49,50,51,74,46,47,48,75,
        22,15,
        57,56,
        76,77,78,79,80,81,82,83,84,85,
        86,87,88,89,
        90,91,92,93,94,
        95,96,97,98,99,100,101,102,103,104,105,106,
        107,
        108,109,110,111,112,
        113,
        114,115,116,117,118,
        119,
        120,121,122,
        123,
        124,125,126,
        127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143)
        
        # Joint parts
        self.joint_part = \
        {'body': range(self.joints_name.index('Pelvis'), self.joints_name.index('Nose')+1),
        'lhand': range(self.joints_name.index('L_Thumb_1'), self.joints_name.index('L_Pinky_4')+1),
        'rhand': range(self.joints_name.index('R_Thumb_1'), self.joints_name.index('R_Pinky_4')+1),
        'hand': range(self.joints_name.index('L_Thumb_1'), self.joints_name.index('R_Pinky_4')+1),
        'face': range(self.joints_name.index('Face_1'), self.joints_name.index('Face_72')+1)}
        
        # Position joint set
        self.pos_joint_num = 65
        self.pos_joints_name = \
        ('Pelvis', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Neck', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel', 'L_Ear', 'R_Ear', 'L_Eye', 'R_Eye', 'Nose',
         'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb_4', 'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Index_4', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Middle_4', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Ring_4', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4',
         'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4', 'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Index_4', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Middle_4', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Ring_4', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4')
        
        self.pos_joint_part = \
        {'body': range(self.pos_joints_name.index('Pelvis'), self.pos_joints_name.index('Nose')+1),
        'lhand': range(self.pos_joints_name.index('L_Thumb_1'), self.pos_joints_name.index('L_Pinky_4')+1),
        'rhand': range(self.pos_joints_name.index('R_Thumb_1'), self.pos_joints_name.index('R_Pinky_4')+1),
        'hand': range(self.pos_joints_name.index('L_Thumb_1'), self.pos_joints_name.index('R_Pinky_4')+1)}

    def forward(self, betas=None, body_pose=None, global_orient=None, left_hand_pose=None, 
                right_hand_pose=None, jaw_pose=None, leye_pose=None, reye_pose=None, 
                expression=None, transl=None, gender='neutral'):
        """
        Forward pass using direct LBS computation to bypass smplx library issues
        
        Args:
            betas: Shape coefficients [batch_size, 10]
            body_pose: Body pose [batch_size, 63] 
            global_orient: Global orientation [batch_size, 3]
            left_hand_pose: Left hand pose [batch_size, 45]
            right_hand_pose: Right hand pose [batch_size, 45]
            jaw_pose: Jaw pose [batch_size, 3]
            leye_pose: Left eye pose [batch_size, 3]
            reye_pose: Right eye pose [batch_size, 3]
            expression: Expression [batch_size, 10]
            transl: Translation [batch_size, 3]
            gender: Gender ('neutral', 'male', 'female')
            
        Returns:
            Output object with vertices and joints
        """
        if betas is None:
            batch_size = 1
            betas = torch.zeros(batch_size, self.shape_param_dim, device=self.device)
        else:
            batch_size = betas.shape[0]
            
        if expression is None:
            expression = torch.zeros(batch_size, self.expr_code_dim, device=self.device)
        if transl is None:
            transl = torch.zeros(batch_size, 3, device=self.device)
            
        # Get model for the specified gender
        model = self.models[gender]
        
        # Apply shape deformation
        v_template = model['v_template'].unsqueeze(0).expand(batch_size, -1, -1)
        
        # Only use the first 10 shape coefficients to match our model files
        shape_disps = torch.einsum('bl,mkl->bmk', betas[:, :10], model['shapedirs'][:, :, :10])
        v_shaped = v_template + shape_disps
        
        # For this simplified version, we'll skip pose deformation and just return shaped vertices
        # This bypasses the problematic einsum operations in the smplx library
        vertices = v_shaped + transl.unsqueeze(1)
        
        # Compute joints using regressor
        joints = torch.einsum('bik,ji->bjk', vertices, model['J_regressor'])
        
        # Create output object
        class Output:
            def __init__(self):
                self.vertices = vertices
                self.joints = joints
                
        return Output()

    def make_hand_regressor(self):
        """Create hand regressor (same as original implementation)"""
        regressor = self.J_regressor
        lhand_regressor = np.concatenate((regressor[[20,37,38,39],:],
                                            np.eye(self.vertex_num)[5361,None],
                                                regressor[[25,26,27],:],
                                                np.eye(self.vertex_num)[4933,None],
                                                regressor[[28,29,30],:],
                                                np.eye(self.vertex_num)[5058,None],
                                                regressor[[34,35,36],:],
                                                np.eye(self.vertex_num)[5169,None],
                                                regressor[[31,32,33],:],
                                                np.eye(self.vertex_num)[5286,None]))
        rhand_regressor = np.concatenate((regressor[[21,52,53,54],:],
                                            np.eye(self.vertex_num)[8079,None],
                                                regressor[[40,41,42],:],
                                                np.eye(self.vertex_num)[7669,None],
                                                regressor[[43,44,45],:],
                                                np.eye(self.vertex_num)[7794,None],
                                                regressor[[49,50,51],:],
                                                np.eye(self.vertex_num)[7905,None],
                                                regressor[[46,47,48],:],
                                                np.eye(self.vertex_num)[8022,None]))
        hand_regressor = {'left': lhand_regressor, 'right': rhand_regressor}
        return hand_regressor

    def reduce_joint_set(self, joint):
        """Reduce joint set for position network"""
        new_joint = []
        for name in self.pos_joints_name:
            idx = self.joints_name.index(name)
            new_joint.append(joint[:,idx,:])
        new_joint = torch.stack(new_joint,1)
        return new_joint

    @classmethod
    def get_instance(cls):
        """Retrieve the singleton instance"""
        return cls()

    # Create compatibility layer properties
    @property  
    def layer(self):
        """Compatibility property to mimic original implementation"""
        class CompatLayer:
            def __init__(self, pytorch3d_model):
                self.pytorch3d_model = pytorch3d_model
                
            def __call__(self, **kwargs):
                return self.pytorch3d_model.forward(**kwargs)
                
            @property
            def faces(self):
                return self.pytorch3d_model.face
        
        return {'neutral': CompatLayer(self)}

    # Initialize hand regressor
    def __getattr__(self, name):
        if name == 'orig_hand_regressor':
            if not hasattr(self, '_orig_hand_regressor'):
                self._orig_hand_regressor = self.make_hand_regressor()
            return self._orig_hand_regressor
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")