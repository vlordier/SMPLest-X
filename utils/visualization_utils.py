import os
import cv2
import numpy as np
import trimesh
# Configure matplotlib for headless rendering before any pyplot imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Mac OpenGL compatibility setup
def setup_mac_opengl():
    """Setup OpenGL for Mac compatibility"""
    # Try multiple OpenGL platforms for Mac compatibility
    platforms = ['osmesa', 'egl', 'glx']
    
    for platform in platforms:
        try:
            os.environ['PYOPENGL_PLATFORM'] = platform
            import pyrender
            print(f"✅ Using OpenGL platform: {platform}")
            return pyrender
        except ImportError as e:
            if "OpenGL" in str(e):
                continue
            else:
                raise
    
    # If all platforms fail, create a mock renderer
    print("⚠️  All OpenGL platforms failed, using mock renderer")
    return create_mock_pyrender()

def create_mock_pyrender():
    """Create mock pyrender for Mac compatibility"""
    from types import ModuleType
    import numpy as np
    
    class MockOffscreenRenderer:
        def __init__(self, viewport_width=640, viewport_height=480):
            self.viewport_width = viewport_width
            self.viewport_height = viewport_height
            
        def render(self, scene):
            # Return placeholder image
            color = np.ones((self.viewport_height, self.viewport_width, 3), dtype=np.uint8) * 128
            depth = np.ones((self.viewport_height, self.viewport_width), dtype=np.float32)
            return color, depth
            
        def delete(self):
            pass
    
    class MockRenderFlags:
        SKIP_CULL_FACES = 1
        SHADOWS_DIRECTIONAL = 2
    
    mock_pyrender = ModuleType('pyrender')
    mock_pyrender.OffscreenRenderer = MockOffscreenRenderer
    mock_pyrender.RenderFlags = MockRenderFlags
    mock_pyrender.PerspectiveCamera = lambda *args, **kwargs: None
    mock_pyrender.IntrinsicsCamera = lambda *args, **kwargs: None
    mock_pyrender.DirectionalLight = lambda: None
    mock_pyrender.Scene = lambda: type('Scene', (), {'add': lambda *args: None})()
    mock_pyrender.Mesh = lambda *args, **kwargs: None
    mock_pyrender.Node = lambda: None
    
    return mock_pyrender

# Try to import pyrender with Mac compatibility
try:
    pyrender = setup_mac_opengl()
except Exception as e:
    print(f"❌ OpenGL setup failed completely: {e}")
    pyrender = create_mock_pyrender()

def vis_keypoints_with_skeleton(img, kps, kps_lines, kp_thresh=0.4, alpha=1):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for line_idx in range(len(kps_lines)):
        i1 = kps_lines[line_idx][0]
        i2 = kps_lines[line_idx][1]
        p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
        p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[line_idx], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[line_idx], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[line_idx], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

def vis_keypoints(img, kps, alpha=1, radius=3, color=None):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    if color is None:
        colors = [cmap(i) for i in np.linspace(0, 1, len(kps) + 2)]
        colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for i in range(len(kps)):
        p = kps[i][0].astype(np.int32), kps[i][1].astype(np.int32)
        if color is None:
            cv2.circle(kp_mask, p, radius=radius, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)
        else:
            cv2.circle(kp_mask, p, radius=radius, color=color, thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

def vis_mesh(img, mesh_vertex, alpha=0.5):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(mesh_vertex))]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    mask = np.copy(img)

    # Draw the mesh
    for i in range(len(mesh_vertex)):
        p = mesh_vertex[i][0].astype(np.int32), mesh_vertex[i][1].astype(np.int32)
        cv2.circle(mask, p, radius=1, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, mask, alpha, 0)

def vis_3d_skeleton(kpt_3d, kpt_3d_vis, kps_lines, filename=None):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]

    for line_idx in range(len(kps_lines)):
        i1 = kps_lines[line_idx][0]
        i2 = kps_lines[line_idx][1]
        x = np.array([kpt_3d[i1,0], kpt_3d[i2,0]])
        y = np.array([kpt_3d[i1,1], kpt_3d[i2,1]])
        z = np.array([kpt_3d[i1,2], kpt_3d[i2,2]])

        if kpt_3d_vis[i1,0] > 0 and kpt_3d_vis[i2,0] > 0:
            ax.plot(x, z, -y, c=colors[line_idx], linewidth=2)
        if kpt_3d_vis[i1,0] > 0:
            ax.scatter(kpt_3d[i1,0], kpt_3d[i1,2], -kpt_3d[i1,1], c=colors[line_idx], marker='o')
        if kpt_3d_vis[i2,0] > 0:
            ax.scatter(kpt_3d[i2,0], kpt_3d[i2,2], -kpt_3d[i2,1], c=colors[line_idx], marker='o')

    # Remove unused variables that reference undefined cfg
    
    if filename is None:
        ax.set_title('3D vis')
    else:
        ax.set_title(filename)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')
    ax.legend()

    plt.show()
    cv2.waitKey(0)

def perspective_projection_robust(vertices_3d, camera_params):
    """
    Project 3D vertices to 2D using camera parameters with robust handling
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
    
    # Draw faces as wireframe - subsample for performance
    faces_to_draw = faces[::max(1, len(faces)//1000)]  # Draw max 1000 faces
    faces_drawn = 0
    
    for face in faces_to_draw:
        # Get the three vertices of the face
        v1_idx, v2_idx, v3_idx = face[0], face[1], face[2]
        
        # Check if all vertices are valid and indices are in range
        if (v1_idx >= len(valid_mask) or v2_idx >= len(valid_mask) or v3_idx >= len(valid_mask)):
            continue
            
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
    
    print(f"Drew {faces_drawn} wireframe faces")
    return img

def render_mesh_improved(img, vertices, faces, camera_params):
    """
    Improved mesh rendering with better Mac compatibility
    """
    print(f"Rendering mesh: {len(vertices)} vertices, {len(faces)} faces")
    
    # Create a copy of the image
    result_img = img.copy()
    
    # Method 1: Draw wireframe
    try:
        result_img = draw_mesh_wireframe(result_img, vertices, faces, camera_params, 
                                       color=(0, 255, 255), thickness=1)
    except Exception as e:
        print(f"Wireframe rendering failed: {e}")
        # Fallback to vertex points
        vertices_2d, valid_mask = perspective_projection_robust(vertices, camera_params)
        vertices_2d_int = vertices_2d.astype(np.int32)
        
        # Draw subset of vertices as points
        step = max(1, len(vertices) // 200)  # Draw ~200 points max
        for i in range(0, len(vertices), step):
            if valid_mask[i]:
                pt = tuple(vertices_2d_int[i])
                if 0 <= pt[0] < img.shape[1] and 0 <= pt[1] < img.shape[0]:
                    cv2.circle(result_img, pt, 1, (0, 255, 255), -1)
    
    return result_img

def save_obj(v, f, file_name='output.obj'):
    obj_file = open(file_name, 'w')
    for i in range(len(v)):
        obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
    for i in range(len(f)):
        obj_file.write('f ' + str(f[i][0]+1) + ' ' + str(f[i][1]+1) + ' ' + str(f[i][2]+1) + '\n') 
    obj_file.close()

def perspective_projection(vertices, cam_param):
    # vertices: [N, 3]
    # cam_param: [3]
    fx, fy= cam_param['focal']
    cx, cy = cam_param['princpt']
    vertices[:, 0] = vertices[:, 0] * fx / vertices[:, 2] + cx
    vertices[:, 1] = vertices[:, 1] * fy / vertices[:, 2] + cy
    return vertices

def render_mesh(img, vertices, faces, cam_param, mesh_as_vertices=False):
    if mesh_as_vertices:
        # to run on cluster where headless pyrender is not supported for A100/V100
        vertices_2d = perspective_projection(vertices, cam_param)
        img = vis_keypoints(img, vertices_2d, alpha=0.8, radius=2, color=(0, 0, 255))
    else:
        focal, princpt = cam_param['focal'], cam_param['princpt']
        # Use PerspectiveCamera for newer pyrender versions
        try:
            camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
        except AttributeError:
            # Fallback to PerspectiveCamera for newer pyrender versions
            camera = pyrender.PerspectiveCamera(yfov=2*np.arctan(princpt[1]/focal[1]), aspectRatio=focal[0]/focal[1])
        # the inverse is same
        pyrender2opencv = np.array([[1.0, 0, 0, 0],
                                    [0, -1, 0, 0],
                                    [0, 0, -1, 0],
                                    [0, 0, 0, 1]])
        

        # render material
        base_color = (1.0, 193/255, 193/255, 1.0)
        material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0,
                alphaMode='OPAQUE',
                baseColorFactor=base_color)
        
        material_new = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.1,
                roughnessFactor=0.4,
                alphaMode='OPAQUE',
                emissiveFactor=(0.2, 0.2, 0.2),
                baseColorFactor=(0.7, 0.7, 0.7, 1))  
        material = material_new
        
        # get body mesh
        body_trimesh = trimesh.Trimesh(vertices, faces, process=False)
        body_mesh = pyrender.Mesh.from_trimesh(body_trimesh, material=material)

        # prepare camera and light
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
        cam_pose = pyrender2opencv @ np.eye(4)
        
        # build scene
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                                        ambient_light=(0.3, 0.3, 0.3))
        scene.add(camera, pose=cam_pose)
        scene.add(light, pose=cam_pose)
        scene.add(body_mesh, 'mesh')

        # render scene
        r = pyrender.OffscreenRenderer(viewport_width=img.shape[1],
                                        viewport_height=img.shape[0],
                                        point_size=1.0)
        
        color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        alpha = 0.8 # set transparency in [0.0, 1.0]

        valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
        valid_mask = valid_mask * alpha
        img = img / 255
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        output_img = (color[:, :, :] * valid_mask + (1 - valid_mask) * img)

        img = (output_img * 255).astype(np.uint8)
    return img