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
from human_models.human_models import SMPLX
from ultralytics import YOLO
from main.base import Tester
from main.config import Config
from utils.data_utils import load_img, process_bbox, generate_patch_image
from utils.visualization_utils import render_mesh
from utils.inference_utils import non_max_suppression
from utils.device_utils import get_device, get_device_name, to_device


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
    """Render mesh without background for isolated mesh visualization"""
    try:
        import pyrender
        import trimesh
        
        # Create mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh = pyrender.Mesh.from_trimesh(mesh)
        
        # Create scene
        scene = pyrender.Scene()
        scene.add(mesh)
        
        # Create camera
        focal, princpt = cam_param['focal'], cam_param['princpt']
        try:
            camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
        except AttributeError:
            # Fallback for newer pyrender versions
            camera = pyrender.PerspectiveCamera(yfov=2*np.arctan(princpt[1]/focal[1]), aspectRatio=focal[0]/focal[1])
        
        scene.add(camera)
        
        # Create light
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        scene.add(light)
        
        # Render
        renderer = pyrender.OffscreenRenderer(img_shape[1], img_shape[0])
        color, _ = renderer.render(scene)
        renderer.delete()
        
        return color
        
    except Exception as e:
        print(f"PyRender failed ({e}), using matplotlib wireframe fallback")
        return render_mesh_matplotlib_fallback(vertices, faces, cam_param, img_shape)


def render_mesh_matplotlib_fallback(vertices, faces, cam_param, img_shape=(512, 512)):
    """Fallback mesh renderer using matplotlib for wireframe visualization"""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
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
        
        # Convert to numpy array
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
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
    parser.add_argument('--num_gpus', type=int, dest='num_gpus')
    parser.add_argument('--file_name', type=str, default='test')
    parser.add_argument('--ckpt_name', type=str, default='model_dump')
    parser.add_argument('--start', type=str, default=1)
    parser.add_argument('--end', type=str, default=1)
    parser.add_argument('--multi_person', action='store_true')
    parser.add_argument('--save_meshes', action='store_true', help='Save 3D meshes in OBJ format')
    parser.add_argument('--save_mesh_renders', action='store_true', help='Save isolated mesh renders without background')
    parser.add_argument('--mesh_format', type=str, default='obj', choices=['obj', 'ply'], help='Mesh file format')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cudnn.benchmark = True

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
    demoer.logger.info(f"Using 1 GPU.")
    demoer.logger.info(f'Inference [{args.file_name}] with [{cfg.model.pretrained_model_path}].')
    demoer._make_model()

    # init detector
    bbox_model = getattr(cfg.inference.detection, "model_path", 
                        './pretrained_models/yolov8x.pt')
    detector = YOLO(bbox_model)

    start = int(args.start)
    end = int(args.end) + 1

    for frame in tqdm(range(start, end)):
        
        # prepare input image
        img_path =osp.join(img_folder, f'{int(frame):06d}.jpg')

        transform = transforms.ToTensor()
        original_img = load_img(img_path)
        vis_img = original_img.copy()
        original_img_height, original_img_width = original_img.shape[:2]
        
        # detection, xyxy
        device_name = get_device_name()
        yolo_bbox = detector.predict(original_img, 
                                device=device_name, 
                                classes=00, 
                                conf=cfg.inference.detection.conf, 
                                save=cfg.inference.detection.save, 
                                verbose=cfg.inference.detection.verbose
                                    )[0].boxes.xyxy.detach().cpu().numpy()

        if len(yolo_bbox)<1:
            # save original image if no bbox
            num_bbox = 0
        if not args.multi_person:
            # only select the largest bbox
            num_bbox = 1
            # yolo_bbox = yolo_bbox[0]
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
            img = to_device(img)[None,:,:,:]
            inputs = {'img': img}
            targets = {}
            meta_info = {}

            # mesh recovery
            with torch.no_grad():
                out = demoer.model(inputs, targets, meta_info, 'test')

            mesh = out['smplx_mesh_cam'].detach().cpu().numpy()[0]
            
            # Save mesh if requested
            frame_name_no_ext = os.path.splitext(os.path.basename(img_path))[0]
            
            if args.save_meshes:
                mesh_filename = f"{frame_name_no_ext}_person{bbox_id}.{args.mesh_format}"
                mesh_path = os.path.join(mesh_folder, mesh_filename)
                
                if args.mesh_format == 'obj':
                    save_mesh_obj(mesh, smpl_x.face, mesh_path)
                elif args.mesh_format == 'ply':
                    save_mesh_ply(mesh, smpl_x.face, mesh_path)
                
                print(f"💾 Saved mesh: {mesh_filename}")
            
            if args.save_mesh_renders:
                mesh_render_filename = f"{frame_name_no_ext}_person{bbox_id}_mesh.jpg"
                mesh_render_path = os.path.join(mesh_render_folder, mesh_render_filename)
                
                # Create camera parameters for isolated mesh rendering
                cam_param = {'focal': [cfg.model.focal[0] / cfg.model.input_body_shape[1] * bbox[2], 
                                     cfg.model.focal[1] / cfg.model.input_body_shape[0] * bbox[3]],
                           'princpt': [cfg.model.princpt[0] / cfg.model.input_body_shape[1] * bbox[2] + bbox[0], 
                                     cfg.model.princpt[1] / cfg.model.input_body_shape[0] * bbox[3] + bbox[1]]}
                
                # Render isolated mesh
                mesh_render = render_mesh_only(mesh, smpl_x.face, cam_param)
                cv2.imwrite(mesh_render_path, mesh_render[:, :, ::-1])  # Convert RGB to BGR for OpenCV
                print(f"🎨 Saved mesh render: {mesh_render_filename}")

            # render mesh
            focal = [cfg.model.focal[0] / cfg.model.input_body_shape[1] * bbox[2], 
                     cfg.model.focal[1] / cfg.model.input_body_shape[0] * bbox[3]]
            princpt = [cfg.model.princpt[0] / cfg.model.input_body_shape[1] * bbox[2] + bbox[0], 
                       cfg.model.princpt[1] / cfg.model.input_body_shape[0] * bbox[3] + bbox[1]]
            
            # draw the bbox on img
            vis_img = cv2.rectangle(vis_img, (int(yolo_bbox[bbox_id][0]), int(yolo_bbox[bbox_id][1])), 
                                    (int(yolo_bbox[bbox_id][2]), int(yolo_bbox[bbox_id][3])), (0, 255, 0), 1)
            # draw mesh with OpenGL fallback for Mac compatibility
            try:
                vis_img = render_mesh(vis_img, mesh, smpl_x.face, {'focal': focal, 'princpt': princpt}, mesh_as_vertices=False)
            except Exception as e:
                if 'OpenGL' in str(e) or 'EGL' in str(e) or 'pyrender' in str(e):
                    print(f"OpenGL rendering failed, using vertex projection fallback: {e}")
                    vis_img = render_mesh(vis_img, mesh, smpl_x.face, {'focal': focal, 'princpt': princpt}, mesh_as_vertices=True)
                else:
                    raise

        # save rendered image
        frame_name = os.path.basename(img_path)
        cv2.imwrite(os.path.join(output_folder, frame_name), vis_img[:, :, ::-1])


if __name__ == "__main__":
    main()
