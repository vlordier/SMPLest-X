.PHONY: clean clean-artifacts clean-outputs clean-meshes clean-renders clean-logs inference help

# Clean all generated artifacts but preserve models and input videos
clean: clean-artifacts
	@echo "✅ Cleaned all artifacts (preserved models and input videos)"

# Clean inference outputs and generated files
clean-artifacts: clean-outputs clean-meshes clean-renders clean-logs
	@echo "🧹 Cleaning inference artifacts..."
	@rm -f *.log debug_output*.log inference_validation.log
	@rm -f *.jpg *.png mesh_overlay_*.jpg comparison_*.jpg mesh_overlay_sample.jpg
	@rm -f *.py.bak *~ .DS_Store
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete 2>/dev/null || true

# Clean output frame directories
clean-outputs:
	@echo "🖼️  Cleaning output frames..."
	@rm -rf demo/output_frames/*/
	@rm -rf outputs/inference_*/

# Clean mesh files
clean-meshes:
	@echo "🔺 Cleaning mesh files..."
	@rm -rf demo/output_meshes/*/

# Clean mesh render files
clean-renders:
	@echo "🎨 Cleaning mesh renders..."
	@rm -rf demo/mesh_renders/*/

# Clean log files
clean-logs:
	@echo "📋 Cleaning logs..."
	@rm -f *.log debug_output*.log inference_validation.log

# Show what will be cleaned (dry run)
show-clean:
	@echo "📁 Files and directories that would be cleaned:"
	@echo "Output frames:"
	@find demo/output_frames/ -type f 2>/dev/null | head -10 || echo "  (none found)"
	@echo "Mesh files:"
	@find demo/output_meshes/ -type f 2>/dev/null | head -10 || echo "  (none found)"
	@echo "Mesh renders:"
	@find demo/mesh_renders/ -type f 2>/dev/null | head -10 || echo "  (none found)"
	@echo "Log files:"
	@find . -maxdepth 1 -name "*.log" -type f 2>/dev/null || echo "  (none found)"
	@echo "Output directories:"
	@find outputs/ -name "inference_*" -type d 2>/dev/null | head -5 || echo "  (none found)"

# Show what will be preserved
show-preserved:
	@echo "📁 Files and directories that will be PRESERVED:"
	@echo "Models:"
	@find pretrained_models/ -name "*.pt" -o -name "*.pth.tar" 2>/dev/null || echo "  (none found)"
	@echo "Input videos:"
	@find demo/ -name "*.mp4" -not -path "*/output_*" -not -name "*comparison*" -not -name "*result*" -not -name "*overlay*" 2>/dev/null || echo "  (none found)"
	@echo "Human model files:"
	@find human_models/human_model_files/ -type f 2>/dev/null | head -5 || echo "  (none found)"

# Run inference with default settings
inference:
	@echo "🚀 Running inference with default settings..."
	@echo "   File: 1349093_720p"
	@echo "   Model: smplest_x_h" 
	@echo "   Frames: 1-30"
	@echo "   Frame skip: 2 (default)"
	@echo "   Saving meshes in PLY format"
	@echo ""
	python main/inference.py \
		--file_name 1349093_720p \
		--ckpt_name smplest_x_h \
		--start 1 \
		--end 30 \
		--save_meshes \
		--mesh_format ply \
		--frame_skip 2

# Run inference with custom parameters
inference-custom:
	@echo "🎯 Running inference with custom parameters..."
	@echo "Usage: make inference-custom FILE=filename MODEL=model START=1 END=30 SKIP=2 FORMAT=ply"
	@echo "Example: make inference-custom FILE=myvideo MODEL=smplest_x_h START=10 END=50 SKIP=3 FORMAT=obj"
	@echo ""
	$(eval FILE ?= 1349093_720p)
	$(eval MODEL ?= smplest_x_h)
	$(eval START ?= 1)
	$(eval END ?= 30)
	$(eval SKIP ?= 2)
	$(eval FORMAT ?= ply)
	python main/inference.py \
		--file_name $(FILE) \
		--ckpt_name $(MODEL) \
		--start $(START) \
		--end $(END) \
		--save_meshes \
		--mesh_format $(FORMAT) \
		--frame_skip $(SKIP)

help:
	@echo "SMPLest-X Makefile Commands:"
	@echo ""
	@echo "Inference:"
	@echo "  make inference         - Run inference with default settings (1349093_720p, frames 1-30, skip=2)"
	@echo "  make inference-custom  - Run inference with custom parameters (see command for usage)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          - Clean all artifacts (output frames, meshes, renders, logs)"
	@echo "  make clean-outputs  - Clean only output frame directories"
	@echo "  make clean-meshes   - Clean only mesh files (.obj, .ply)"
	@echo "  make clean-renders  - Clean only mesh render images"
	@echo "  make clean-logs     - Clean only log files"
	@echo ""
	@echo "Information:"
	@echo "  make show-clean     - Show what files would be cleaned (dry run)"
	@echo "  make show-preserved - Show what files will be preserved"
	@echo "  make help           - Show this help message"
	@echo ""
	@echo "Note: Models in pretrained_models/ and input videos in demo/ are preserved."