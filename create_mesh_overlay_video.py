#!/usr/bin/env python3
"""
Create a video with 3D mesh overlays on the original video frames.
This script combines the inference output frames (which have mesh rendering) into a video.
"""

import cv2
import numpy as np
from pathlib import Path
import os
import subprocess

def create_mesh_overlay_video(
    input_frames_dir,
    output_video_path,
    fps=30,
    video_quality='high'
):
    """
    Create video from frames with mesh overlays
    
    Args:
        input_frames_dir: Directory containing frames with mesh overlays
        output_video_path: Path for output video
        fps: Frames per second
        video_quality: 'high', 'medium', or 'low'
    """
    
    frames_dir = Path(input_frames_dir)
    
    if not frames_dir.exists():
        print(f"❌ Frames directory not found: {frames_dir}")
        return False
    
    # Get all frame files
    frame_files = sorted([f for f in frames_dir.glob("*.jpg") if f.name.startswith("0000")])
    
    if not frame_files:
        print(f"❌ No frame files found in {frames_dir}")
        return False
    
    print("🎬 Creating mesh overlay video...")
    print(f"   Input frames: {len(frame_files)} files")
    print(f"   Output: {output_video_path}")
    print(f"   FPS: {fps}")
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(str(frame_files[0]))
    if first_frame is None:
        print(f"❌ Could not read first frame: {frame_files[0]}")
        return False
    
    height, width = first_frame.shape[:2]
    print(f"   Resolution: {width}x{height}")
    
    # Set up video quality parameters
    quality_settings = {
        'high': {
            'crf': '18',
            'preset': 'slow',
            'profile': 'high'
        },
        'medium': {
            'crf': '23', 
            'preset': 'medium',
            'profile': 'main'
        },
        'low': {
            'crf': '28',
            'preset': 'fast', 
            'profile': 'baseline'
        }
    }
    
    settings = quality_settings.get(video_quality, quality_settings['medium'])
    
    # Create temporary file list for ffmpeg
    temp_list_file = "temp_frame_list.txt"
    
    try:
        # Create frame list file for ffmpeg
        with open(temp_list_file, 'w') as f:
            for frame_file in frame_files:
                f.write(f"file '{frame_file.absolute()}'\n")
        
        # Use ffmpeg to create high-quality video
        ffmpeg_cmd = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', temp_list_file,
            '-c:v', 'libx264',
            '-crf', settings['crf'],
            '-preset', settings['preset'],
            '-profile:v', settings['profile'],
            '-pix_fmt', 'yuv420p',
            '-r', str(fps),
            '-y',  # Overwrite output file
            str(output_video_path)
        ]
        
        print("🔧 Running ffmpeg...")
        print(f"   Command: {' '.join(ffmpeg_cmd[:8])}...")
        
        result = subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print("✅ Video created successfully!")
            
            # Check output file
            output_path = Path(output_video_path)
            if output_path.exists():
                file_size = output_path.stat().st_size / (1024 * 1024)  # MB
                print(f"   File size: {file_size:.1f} MB")
                return True
            else:
                print("❌ Output file not found after ffmpeg completion")
                return False
                
        else:
            print(f"❌ ffmpeg failed with return code {result.returncode}")
            print(f"   Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ ffmpeg timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Error creating video: {e}")
        return False
    finally:
        # Clean up temporary file
        if os.path.exists(temp_list_file):
            os.remove(temp_list_file)

def create_side_by_side_comparison(
    original_video_path,
    mesh_frames_dir,
    output_video_path,
    fps=30
):
    """
    Create side-by-side comparison video: original | mesh overlay
    """
    
    print("🎬 Creating side-by-side comparison video...")
    
    # Extract frames from original video
    temp_original_dir = "temp_original_frames"
    os.makedirs(temp_original_dir, exist_ok=True)
    
    try:
        # Extract original frames
        extract_cmd = [
            'ffmpeg',
            '-i', str(original_video_path),
            '-r', str(fps),
            '-q:v', '2',  # High quality
            f'{temp_original_dir}/frame_%06d.jpg',
            '-y'
        ]
        
        print("🔧 Extracting original video frames...")
        result = subprocess.run(extract_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Failed to extract frames: {result.stderr}")
            return False
        
        # Get frame files
        original_frames = sorted(Path(temp_original_dir).glob("*.jpg"))
        mesh_frames = sorted(Path(mesh_frames_dir).glob("*.jpg"))
        
        if not original_frames or not mesh_frames:
            print("❌ Missing frames for comparison")
            return False
        
        print(f"   Original frames: {len(original_frames)}")
        print(f"   Mesh frames: {len(mesh_frames)}")
        
        # Create side-by-side frames
        temp_combined_dir = "temp_combined_frames"
        os.makedirs(temp_combined_dir, exist_ok=True)
        
        num_frames = min(len(original_frames), len(mesh_frames))
        
        for i in range(num_frames):
            # Read frames
            orig_frame = cv2.imread(str(original_frames[i]))
            mesh_frame = cv2.imread(str(mesh_frames[i]))
            
            if orig_frame is None or mesh_frame is None:
                continue
            
            # Resize frames to same height
            height = min(orig_frame.shape[0], mesh_frame.shape[0])
            orig_frame = cv2.resize(orig_frame, (int(orig_frame.shape[1] * height / orig_frame.shape[0]), height))
            mesh_frame = cv2.resize(mesh_frame, (int(mesh_frame.shape[1] * height / mesh_frame.shape[0]), height))
            
            # Combine side by side
            combined = np.hstack([orig_frame, mesh_frame])
            
            # Add labels
            cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined, "With Mesh Overlay", (orig_frame.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            output_frame_path = f"{temp_combined_dir}/combined_{i:06d}.jpg"
            cv2.imwrite(output_frame_path, combined)
        
        # Create video from combined frames
        create_cmd = [
            'ffmpeg',
            '-framerate', str(fps),
            '-i', f'{temp_combined_dir}/combined_%06d.jpg',
            '-c:v', 'libx264',
            '-crf', '20',
            '-pix_fmt', 'yuv420p',
            '-y',
            str(output_video_path)
        ]
        
        print("🔧 Creating comparison video...")
        result = subprocess.run(create_cmd, capture_output=True, text=True)
        
        success = result.returncode == 0
        if success:
            print("✅ Comparison video created!")
        else:
            print(f"❌ Failed to create video: {result.stderr}")
        
        return success
        
    finally:
        # Clean up
        import shutil
        if os.path.exists(temp_original_dir):
            shutil.rmtree(temp_original_dir)
        if os.path.exists(temp_combined_dir):
            shutil.rmtree(temp_combined_dir)

def main():
    # Configuration
    project_dir = Path("/Users/vincent.lordier/Work/SMPLest-X")
    video_name = "1349093_720p"
    
    # Paths
    mesh_frames_dir = project_dir / "demo" / "output_frames" / video_name
    original_video_path = project_dir / "demo" / f"{video_name}.mp4"
    
    # Output paths
    mesh_overlay_video = project_dir / "demo" / f"mesh_overlay_{video_name}.mp4"
    comparison_video = project_dir / "demo" / f"comparison_{video_name}.mp4"
    
    print("🎬 MESH OVERLAY VIDEO GENERATOR")
    print("=" * 50)
    
    # Check if we have output frames
    if not mesh_frames_dir.exists():
        print(f"❌ Mesh frames directory not found: {mesh_frames_dir}")
        print("   Run inference first to generate frames with mesh overlays")
        return
    
    frame_files = list(mesh_frames_dir.glob("*.jpg"))
    if not frame_files:
        print(f"❌ No output frames found in: {mesh_frames_dir}")
        return
        
    print(f"📁 Found {len(frame_files)} output frames with mesh overlays")
    
    # Method 1: Create video from mesh overlay frames
    print("\n🎥 Creating mesh overlay video...")
    success1 = create_mesh_overlay_video(
        mesh_frames_dir,
        mesh_overlay_video,
        fps=30,
        video_quality='high'
    )
    
    if success1:
        print(f"✅ Mesh overlay video saved: {mesh_overlay_video}")
    
    # Method 2: Create side-by-side comparison (if original video exists)
    if original_video_path.exists():
        print("\n🎥 Creating side-by-side comparison video...")
        success2 = create_side_by_side_comparison(
            original_video_path,
            mesh_frames_dir,
            comparison_video,
            fps=30
        )
        
        if success2:
            print(f"✅ Comparison video saved: {comparison_video}")
    else:
        print(f"⚠️  Original video not found: {original_video_path}")
        print("   Skipping side-by-side comparison")
    
    print("\n" + "=" * 50)
    if success1:
        print("🎉 SUCCESS: Mesh overlay video generated!")
        print(f"📹 Video file: {mesh_overlay_video}")
        
        # Show file info
        if mesh_overlay_video.exists():
            file_size = mesh_overlay_video.stat().st_size / (1024 * 1024)
            print(f"📊 File size: {file_size:.1f} MB")
    else:
        print("❌ Failed to generate mesh overlay video")

if __name__ == "__main__":
    main()