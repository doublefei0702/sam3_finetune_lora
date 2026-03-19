# SAM3 Fine-Tuning Guide for Custom Dataset

This guide provides complete instructions for fine-tuning SAM3 on your custom dataset using LoRA (Low-Rank Adaptation) with the provided data and checkpoint.

## Prerequisites

### 1. Environment Setup
```bash
# Navigate to the SAM3 LoRA directory
cd /root/autodl-tmp/SAM3_LoRA

# Install the package
pip install -e .

# Verify CUDA and PyTorch
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, PyTorch version: {torch.__version__}')"
```

### 2. Data Structure Verification
Your data is already properly structured in COCO format:

```
/root/autodl-tmp/sam3_finetune_dataset/data/
├── train/
│   ├── _annotations.coco.json    # Training annotations
│   ├── rgb_0000.png             # Training images
│   ├── rgb_0001.png
│   └── ...
├── valid/
│   ├── _annotations.coco.json    # Validation annotations
│   ├── rgb_0004.png             # Validation images
│   ├── rgb_0005.png
│   └── ...
```

### 3. Checkpoint Verification
Your SAM3 checkpoint is ready at:
- **Path**: `/root/autodl-tmp/sam3_checkpoint/sam3.pt`
- **Size**: ~3.4GB
- **Format**: PyTorch model file
**在train_sam3_lora_native.py中修改：**
```
        self.model = build_sam3_image_model(
            device=self.device.type,
            compile=False,
            checkpoint_path="/root/autodl-tmp/sam3_checkpoint/sam3.pt",  # 设置直接加载权重
            load_from_HF=False,  # Tries to download from HF if checkpoint_path is None
            bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
            eval_mode=False
        )
```
## Configuration

### Modified Configuration File

A custom configuration file has been created at `/root/autodl-tmp/SAM3_LoRA/configs/sam3_finetune_config.yaml` with the following key settings:

**Key Changes from Default:**
- `data_dir: "/root/autodl-tmp/sam3_finetune_dataset/data"` - Points to your data
- `batch_size: 2` - Conservative for 32GB GPU
- `rank: 16` - Good balance of performance and memory
- `apply_to_detr_encoder: false` - Disabled to save memory
- `apply_to_detr_decoder: false` - Disabled to save memory
- `gradient_accumulation_steps: 4` - Effective batch size of 8

## Training Process

### 1. Single GPU Training (Recommended for RTX 5090 32GB)

```bash
# Navigate to the SAM3 LoRA directory
cd /root/autodl-tmp/SAM3_LoRA

# Start training with your custom configuration
python train_sam3_lora_native.py --config configs/sam3_finetune_config.yaml
```

**Expected Output:**
```
Building SAM3 model...
Applying LoRA...
Applied LoRA to 64 modules
Trainable params: 5,898,240 (0.69%)

Loading training data from /root/autodl-tmp/sam3_finetune_dataset/data...
Loaded COCO dataset: train split
  Images: [number]
  Annotations: [number]
  Categories: {category mappings}

Loading validation data from /root/autodl-tmp/sam3_finetune_dataset/data...
Loaded COCO dataset: valid split
  Images: [number]
  Annotations: [number]

Starting training for 100 epochs...
Training samples: [X], Validation samples: [Y]

Epoch 1: 100%|████████| [batches]/[batches] [time<00:00, loss=XXX]
Validation: 100%|████████| [batches]/[batches] [time<00:00, val_loss=XXX]

Epoch 1/100 - Train Loss: XXX.XXXXXX, Val Loss: XXX.XXXXXX
✓ New best model saved (val_loss: XXX.XXXXXX)
...
```

### 2. Multi-GPU Training (Optional)

If you want to use multiple GPUs:

```bash
# 2 GPU training
python train_sam3_lora_native.py --config configs/sam3_finetune_config.yaml --device 0 1

# 4 GPU training
python train_sam3_lora_native.py --config configs/sam3_finetune_config.yaml --device 0 1 2 3
```

**Note**: The script automatically handles distributed training setup when multiple GPUs are specified.

## Training Tips

### Memory Management
- **RTX 5090 32GB**: Should handle the configuration well
- **Monitor GPU memory**: Use `nvidia-smi` to check memory usage
- **If OOM occurs**: Reduce `batch_size` to 1 or `rank` to 8

### Training Duration
- **Estimated time**: 2-6 hours depending on dataset size
- **Checkpoints saved**: Every 50 steps
- **Best model**: Saved based on validation loss

### Expected Loss Values
- **Initial loss**: 100-200 (normal for SAM3)
- **Final loss**: 10-50 (good fine-tuning)
- **Validation loss**: Should track training loss

## After Training

### Model Checkpoints
Training saves two models in `/root/autodl-tmp/SAM3_LoRA/outputs/sam3_finetune/`:

1. **`best_lora_weights.pt`** - Best model based on validation loss
2. **`last_lora_weights.pt`** - Model from the last epoch

### Validation (Optional)
To compute full metrics (mAP, cgF1) after training:

```bash
# Navigate to the SAM3 LoRA directory
cd /root/autodl-tmp/SAM3_LoRA

# Run validation on validation set
python validate_sam3_lora.py \
  --config configs/sam3_finetune_config.yaml \
  --weights outputs/sam3_finetune/best_lora_weights.pt \
  --val_data_dir /root/autodl-tmp/sam3_finetune_dataset/data/valid

# Run validation on test set (if you create one)
python validate_sam3_lora.py \
  --config configs/sam3_finetune_config.yaml \
  --weights outputs/sam3_finetune/best_lora_weights.pt \
  --val_data_dir /root/autodl-tmp/sam3_finetune_dataset/data/test
```

## Inference

### Basic Inference
```bash
# Navigate to the SAM3 LoRA directory
cd /root/autodl-tmp/SAM3_LoRA

# Basic inference
python infer_sam.py \
  --config configs/sam3_finetune_config.yaml \
  --weights outputs/sam3_finetune/best_lora_weights.pt \
  --image path/to/your/image.jpg \
  --output predictions.png

# With text prompt
python infer_sam.py \
  --config configs/sam3_finetune_config.yaml \
  --weights outputs/sam3_finetune/best_lora_weights.pt \
  --image path/to/your/image.jpg \
  --prompt "your object description" \
  --output predictions.png
```

### Multiple Prompts
```bash
# Detect multiple object types
python infer_sam.py \
  --config configs/sam3_finetune_config.yaml \
  --weights outputs/sam3_finetune/best_lora_weights.pt \
  --image path/to/your/image.jpg \
  --prompt "object1" "object2" "object3" \
  --output predictions.png
```

### NMS Filtering
```bash
# Reduce overlapping boxes
python infer_sam.py \
  --config configs/sam3_finetune_config.yaml \
  --weights outputs/sam3_finetune/best_lora_weights.pt \
  --image path/to/your/image.jpg \
  --prompt "object" \
  --nms-iou 0.3 \
  --output clean_predictions.png
```

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**
   ```bash
   # Reduce batch size
   # Edit configs/sam3_finetune_config.yaml:
   # batch_size: 1
   # rank: 8
   ```

2. **Slow Training**
   ```bash
   # Increase num_workers
   # Edit configs/sam3_finetune_config.yaml:
   # num_workers: 4
   ```

3. **High Loss**
   ```bash
   # Increase learning rate slightly
   # Edit configs/sam3_finetune_config.yaml:
   # learning_rate: 1e-4
   ```

4. **Overfitting**
   ```bash
   # Reduce rank or add more dropout
   # Edit configs/sam3_finetune_config.yaml:
   # rank: 8
   # dropout: 0.2
   ```

### Verification Commands

```bash
# Check data structure
ls -la /root/autodl-tmp/sam3_finetune_dataset/data/train/
ls -la /root/autodl-tmp/sam3_finetune_dataset/data/valid/

# Check checkpoint
ls -lh /root/autodl-tmp/sam3_checkpoint/sam3.pt

# Check configuration
cat configs/sam3_finetune_config.yaml

# Test model loading
python -c "
from train_sam3_lora_native import SAM3TrainerNative
trainer = SAM3TrainerNative('configs/sam3_finetune_config.yaml')
print('Model loaded successfully')
"
```

## Performance Optimization

### For RTX 5090 32GB

The provided configuration is optimized for your hardware:

- **Memory usage**: ~15-20GB during training
- **Training speed**: ~2-5 images/second
- **Checkpoint size**: ~10-20MB (LoRA weights only)

### Advanced Optimization

If you want to push performance further:

```yaml
# In configs/sam3_finetune_config.yaml
training:
  mixed_precision: "bf16"    # Already enabled
  use_compile: true          # Add this for faster training
  batch_size: 4              # Increase if memory allows
  gradient_accumulation_steps: 2  # Adjust accordingly
```

## Expected Results

### Training Metrics
- **Train Loss**: Should decrease from 100-200 to 10-50
- **Validation Loss**: Should follow similar trend
- **Training Time**: 2-6 hours for 100 epochs

### Model Size
- **LoRA weights**: 10-20MB (tiny compared to full model)
- **Inference speed**: Same as original SAM3
- **Memory usage**: ~2-4GB during inference

### Quality
- **Segmentation accuracy**: Should improve on your specific dataset
- **Text understanding**: Enhanced for your domain vocabulary
- **Generalization**: Maintains SAM3's zero-shot capabilities

## Next Steps

1. **Start training**: Run the training command above
2. **Monitor progress**: Check loss values and GPU usage
3. **Validate results**: Use the validation script after training
4. **Test inference**: Try the inference commands on your images
5. **Iterate**: Adjust hyperparameters if needed

The configuration provided should work well out-of-the-box for your RTX 5090 32GB setup with the provided dataset and checkpoint. Happy fine-tuning!