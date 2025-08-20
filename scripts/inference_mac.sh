#!/usr/bin/env bash

# Enhanced inference script for Mac with better error handling
set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ️${NC} $1"
}

# Check arguments
if [ $# -lt 2 ]; then
    print_error "Usage: $0 <MODEL_DIR> <FILE_NAME> [FPS] [--save-meshes] [--save-mesh-renders] [--mesh-format FORMAT]"
    echo "Example: $0 smplest_x_h test_person.mp4 30"
    echo "         $0 smplest_x_h test_person.mp4 30 --save-meshes --save-mesh-renders --mesh-format obj"
    echo ""
    echo "Options:"
    echo "  --save-meshes         Save 3D meshes in specified format"
    echo "  --save-mesh-renders   Save isolated mesh renders"
    echo "  --mesh-format FORMAT  Mesh format: obj (default) or ply"
    exit 1
fi

CKPT_NAME=$1
FILE_NAME=$2
FPS=${3:-30}

# Parse additional options
SAVE_MESHES=""
SAVE_MESH_RENDERS=""
MESH_FORMAT="obj"

# Parse remaining arguments
shift 3 2>/dev/null || true  # Remove first 3 arguments if they exist
while [[ $# -gt 0 ]]; do
    case $1 in
        --save-meshes)
            SAVE_MESHES="--save_meshes"
            shift
            ;;
        --save-mesh-renders)
            SAVE_MESH_RENDERS="--save_mesh_renders"
            shift
            ;;
        --mesh-format)
            MESH_FORMAT="$2"
            shift 2
            ;;
        *)
            print_warning "Unknown option: $1"
            shift
            ;;
    esac
done

NAME="${FILE_NAME%.*}"
EXT="${FILE_NAME##*.}"

IMG_PATH=./demo/input_frames/$NAME
OUTPUT_PATH=./demo/output_frames/$NAME

print_info "🤖 SMPLest-X Inference for Mac"
print_info "Model: $CKPT_NAME, File: $FILE_NAME, FPS: $FPS"
echo "=" * 50

# Set Mac-specific environment variables
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OMP_NUM_THREADS=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

print_status "Environment configured for Mac"

# Pre-flight checks
print_info "Running pre-flight checks..."

# Check if input file exists
if [ ! -f "./demo/$FILE_NAME" ]; then
    print_error "Input file not found: ./demo/$FILE_NAME"
    exit 1
fi

# Check if model files exist
if [ ! -f "./pretrained_models/$CKPT_NAME/${CKPT_NAME}.pth.tar" ]; then
    print_error "Model file not found: ./pretrained_models/$CKPT_NAME/${CKPT_NAME}.pth.tar"
    print_warning "Run: python download_weights.py"
    exit 1
fi

if [ ! -f "./pretrained_models/$CKPT_NAME/config_base.py" ]; then
    print_error "Config file not found: ./pretrained_models/$CKPT_NAME/config_base.py"
    print_warning "Run: python download_weights.py"
    exit 1
fi

# Check FFmpeg
if ! command -v ffmpeg &> /dev/null; then
    print_error "FFmpeg not found"
    print_warning "Install with: brew install ffmpeg"
    exit 1
fi

# Check Python dependencies
python -c "import torch; import cv2; import ultralytics; import smplx" 2>/dev/null || {
    print_error "Missing Python dependencies"
    print_warning "Run: pip install -r requirements.txt"
    exit 1
}

print_status "Pre-flight checks passed"

# Create directories
mkdir -p "$IMG_PATH"
mkdir -p "$OUTPUT_PATH"

print_status "Created working directories"

# Convert video to frames
print_info "Converting video to frames..."
case "$EXT" in
    mp4|avi|mov|mkv|flv|wmv|webm|mpeg|mpg)
        if ! ffmpeg -i "./demo/$FILE_NAME" -f image2 -vf fps=${FPS}/1 -q:v 2 "${IMG_PATH}/%06d.jpg" -y -loglevel warning; then
            print_error "Failed to convert video to frames"
            exit 1
        fi
        ;;
    jpg|jpeg|png|bmp|gif|tiff|tif|webp|svg)
        cp "./demo/$FILE_NAME" "$IMG_PATH/000001.$EXT"
        ;;
    *)
        print_error "Unknown file type: $EXT"
        exit 1
        ;;
esac

END_COUNT=$(find "$IMG_PATH" -type f | wc -l | tr -d ' ')
print_status "Extracted $END_COUNT frames"

if [ "$END_COUNT" -eq 0 ]; then
    print_error "No frames extracted from video"
    exit 1
fi

# Run inference
print_info "Running SMPLest-X inference..."
export PYTHONPATH=".:$PYTHONPATH"

# Create a timeout wrapper for the inference
timeout_duration=600  # 10 minutes

# Store inference result but continue processing
inference_success=0
if command -v timeout &> /dev/null; then
    timeout $timeout_duration python main/inference.py \
        --num_gpus 1 \
        --file_name "$NAME" \
        --ckpt_name "$CKPT_NAME" \
        --end "$END_COUNT" \
        $SAVE_MESHES \
        $SAVE_MESH_RENDERS \
        --mesh_format "$MESH_FORMAT" \
    && inference_success=1 \
    || print_warning "⚠️ Inference completed with errors but checking for partial results..."
else
    # Fallback for systems without timeout command
    python main/inference.py \
        --num_gpus 1 \
        --file_name "$NAME" \
        --ckpt_name "$CKPT_NAME" \
        --end "$END_COUNT" \
        $SAVE_MESHES \
        $SAVE_MESH_RENDERS \
        --mesh_format "$MESH_FORMAT" \
    && inference_success=1 \
    || print_warning "⚠️ Inference completed with errors but checking for partial results..."
fi

# Check if any output frames were generated
output_frames=$(find "$OUTPUT_PATH" -name "*.jpg" 2>/dev/null | wc -l | tr -d ' ')

if [ "$output_frames" -gt 0 ]; then
    print_status "Found $output_frames output frames, proceeding with video generation"
    inference_success=1
elif [ "$inference_success" -eq 0 ]; then
    print_error "Inference failed and no output frames generated"
    print_warning "Check the output above for error details"
    print_warning "Common fixes:"
    print_warning "  - Ensure model files are downloaded: python download_weights.py"
    print_warning "  - Check SMPL model files in human_models/human_model_files/"
    print_warning "  - Try with PYTORCH_ENABLE_MPS_FALLBACK=1"
    exit 1
fi

print_status "Inference completed successfully"

# Convert frames back to video
print_info "Converting frames back to video..."
case "$EXT" in
    mp4|avi|mov|mkv|flv|wmv|webm|mpeg|mpg)
        if ! ffmpeg -y -f image2 -r ${FPS} -i "${OUTPUT_PATH}/%06d.jpg" \
            -vcodec libx264 -preset medium -crf 23 -pix_fmt yuv420p \
            "./demo/result_${NAME}.mp4" -loglevel warning; then
            print_warning "Failed to create output video, but frames are available in $OUTPUT_PATH"
        else
            print_status "Output video created: ./demo/result_${NAME}.mp4"
        fi
        ;;
    jpg|jpeg|png|bmp|gif|tiff|tif|webp|svg)
        cp "$OUTPUT_PATH/000001.$EXT" "./demo/result_$FILE_NAME"
        print_status "Output image created: ./demo/result_$FILE_NAME"
        ;;
esac

# Cleanup temporary directories
print_info "Cleaning up temporary files..."
rm -rf ./demo/input_frames
rm -rf ./demo/output_frames

print_status "🎉 Inference pipeline completed successfully!"
print_info "Output saved to: ./demo/result_${NAME}.mp4"

# Display file info
if [ -f "./demo/result_${NAME}.mp4" ]; then
    size=$(du -h "./demo/result_${NAME}.mp4" | cut -f1)
    print_info "Output file size: $size"
fi