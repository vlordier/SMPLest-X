#!/usr/bin/env python3
"""
Create overlay frames by combining original video frames with mesh renders.
This creates frames with mesh overlays suitable for video generation.
"""

import cv2
import numpy as np
from pathlib import Path
import os
import json

def extract_frame_number(filename):
    """Extract frame number from filename like '000001_person0_mesh.jpg'"""
    return int(filename.split('_')[0])

def create_overlay_frames(
    input_video_path,
    mesh_renders_dir, 
    output_dir,
    fps=25,
    frame_skip=2,
    alpha=0.7  # Overlay transparency
):
    """
    Create overlay frames by combining original video frames with mesh renders
    
    Args:
        input_video_path: Path to original video
        mesh_renders_dir: Directory containing mesh render images
        output_dir: Directory to save overlay frames
        fps: Original video FPS
        frame_skip: Frame skip used during inference
        alpha: Overlay transparency (0.0=transparent, 1.0=opaque)
    """
    
    input_video = Path(input_video_path)
    renders_dir = Path(mesh_renders_dir) 
    output_path = Path(output_dir)
    
    if not input_video.exists():
        print(f"❌ Input video not found: {input_video}")
        return False
    
    if not renders_dir.exists():
        print(f"❌ Mesh renders directory not found: {renders_dir}")
        return False
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get mesh render files
    render_files = sorted([f for f in renders_dir.glob("*_mesh.jpg")])
    if not render_files:
        print(f"❌ No mesh render files found in {renders_dir}")
        return False
    
    print("🎬 Creating overlay frames...")
    print(f"   Input video: {input_video}")
    print(f"   Mesh renders: {len(render_files)} files") 
    print(f"   Output: {output_path}")
    print(f"   Alpha: {alpha}")
    
    # Open video
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        print(f"❌ Could not open video: {input_video}")
        return False
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"   Video frames: {total_frames}")
    
    success_count = 0
    
    # Process each mesh render
    for render_file in render_files:
        try:
            # Extract frame number from mesh render filename
            frame_num = extract_frame_number(render_file.name)
            
            # Set video to correct frame (0-indexed)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
            ret, original_frame = cap.read()
            
            if not ret:
                print(f"⚠️  Could not read frame {frame_num} from video")
                continue
            
            # Read mesh render
            mesh_render = cv2.imread(str(render_file))
            if mesh_render is None:
                print(f"⚠️  Could not read mesh render: {render_file}")
                continue
            
            # Get dimensions
            orig_h, orig_w = original_frame.shape[:2]
            render_h, render_w = mesh_render.shape[:2]
            
            # Resize mesh render to match original frame
            mesh_render_resized = cv2.resize(mesh_render, (orig_w, orig_h))
            
            # Create overlay by blending
            # Convert mesh render to have alpha channel for non-black pixels
            mesh_mask = np.any(mesh_render_resized != [0, 0, 0], axis=2)
            
            # Create blended image
            overlay_frame = original_frame.copy()
            
            # Only overlay where mesh render has non-black pixels
            if np.any(mesh_mask):
                # Blend the mesh render onto original frame
                overlay_frame[mesh_mask] = cv2.addWeighted(
                    original_frame[mesh_mask], 
                    1 - alpha,
                    mesh_render_resized[mesh_mask], 
                    alpha, 
                    0
                )
            
            # Save overlay frame
            output_filename = f"{frame_num:06d}.jpg"
            output_filepath = output_path / output_filename
            
            success = cv2.imwrite(str(output_filepath), overlay_frame)
            if success:
                success_count += 1
                print(f"✅ Created overlay: {output_filename}")
            else:
                print(f"❌ Failed to save: {output_filename}")
                
        except Exception as e:
            print(f"❌ Error processing {render_file}: {e}")
            continue
    
    cap.release()
    
    print(f"\n📊 Summary:")
    print(f"   Overlay frames created: {success_count}/{len(render_files)}")
    print(f"   Output directory: {output_path}")
    
    return success_count > 0

def main():
    # Configuration
    project_dir = Path("/Users/vincent.lordier/Work/SMPLest-X")
    video_name = "1349093_720p"
    
    # Paths
    input_video = project_dir / "demo" / f"{video_name}.mp4"
    mesh_renders_dir = project_dir / "demo" / "mesh_renders" / video_name
    output_dir = project_dir / "demo" / "overlay_frames" / video_name
    inference_summary_path = project_dir / "demo" / "output_frames" / video_name / "inference_summary.json"
    
    print("🎨 MESH OVERLAY FRAME CREATOR")
    print("=" * 50)
    
    # Load timing parameters
    fps = 25
    frame_skip = 2
    
    if inference_summary_path.exists():
        try:
            with open(inference_summary_path) as f:
                summary = json.load(f)
                fps = summary.get('config', {}).get('fps', fps)
                frame_skip = summary.get('config', {}).get('frame_skip', frame_skip)
                
            print(f"📊 Detected inference parameters:")
            print(f"   Original FPS: {fps}")
            print(f"   Frame skip: {frame_skip}")
        except Exception as e:
            print(f"⚠️  Could not read inference summary: {e}")
    
    # Create overlay frames
    success = create_overlay_frames(
        input_video,
        mesh_renders_dir,
        output_dir,
        fps=fps,
        frame_skip=frame_skip,
        alpha=0.6  # 60% mesh render, 40% original
    )
    
    if success:
        print("\n🎉 SUCCESS: Overlay frames created!")
        print(f"📁 Directory: {output_dir}")
        
        # Count files
        overlay_files = list(output_dir.glob("*.jpg"))
        print(f"📊 Files: {len(overlay_files)} overlay frames")
        
        # Now we can create video with correct timing
        print(f"\n💡 Next steps:")
        print(f"   Run create_mesh_overlay_video.py to create video")
        print(f"   The overlay frames have mesh renders blended with original video")
        
    else:
        print("❌ Failed to create overlay frames")

if __name__ == "__main__":
    main()