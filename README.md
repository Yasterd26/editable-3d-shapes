# Editable 3D Shape Priors

An interactive web application for editing 3D shapes using DeepSDF neural networks. This project enables semantic control over mugs and chairs through learned latent representations.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Features

- **Interactive 3D Editing**: Real-time mesh manipulation with semantic controls
- **Semantic Sliders**: Edit specific attributes like "Handle Length", "Leg Height", "Seat Width"
- **Multiple Categories**: Support for Mugs and Chairs with optimized models
- **Mesh Upload**: Upload custom shapes and edit them in the learned latent space
- **High Quality**: 128³ resolution with Laplacian smoothing for detailed meshes

## Demo

The application provides an intuitive Gradio interface where you can:
1. Select a category (Mug or Chair)
2. Load the trained model
3. Use semantic sliders to control shape attributes
4. Randomize base shapes from the training set
5. Upload custom meshes for editing

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional, but recommended for training)

### Setup

```bash
# Clone the repository
git clone https://github.com/Yasterd26/editable-3d-shapes.git
cd editable-3d-shapes

# Install dependencies
pip install torch torchvision torchaudio
pip install gradio trimesh scikit-image scikit-learn numpy
pip install python-docx  # For report generation
```

## Quick Start

### Running the Web App

```bash
python app.py
```

Open your browser to `http://localhost:7860`

### Model Checkpoints

The app automatically loads the latest trained checkpoints:
- **Mugs**: `models/checkpoints/model_epoch_*.pth` (hidden_dim=512)
- **Chairs**: `models/checkpoints/chairs/model_epoch_*.pth` (hidden_dim=256)

> **Note**: Model checkpoints are not included in the repository due to size. You need to train models or download pre-trained weights.

## Training

### 1. Data Preparation

Download ShapeNet dataset and preprocess into SDF samples:

```bash
# Preprocess GLB meshes to SDF samples
python utils/preprocess_sdf.py \
  --input_dir data/shapenet_mugs/mug \
  --output_dir data/processed/mugs
```

### 2. Train DeepSDF Model

```bash
# Train Mug model (512 hidden dims for quality)
python train.py \
  --data_dir data/processed/mugs \
  --save_dir models/checkpoints \
  --epochs 300 \
  --hidden_dim 512 \
  --batch_size 32

# Train Chair model
python train.py \
  --data_dir data/processed/chairs \
  --save_dir models/checkpoints/chairs \
  --epochs 300 \
  --hidden_dim 256 \
  --batch_size 32
```

**Training Tips:**
- Use `--hidden_dim 512` for mugs (better quality)
- Use `--hidden_dim 256` for chairs (faster, sufficient)
- Checkpoint saved every 10 epochs (configurable with `--save_freq`)

## Project Structure

```
editable-3d-shapes/
├── app.py                      # Gradio web interface
├── train.py                    # DeepSDF training script
├── evaluate.py                 # Mesh reconstruction utilities
├── models/
│   └── deep_sdf.py            # DeepSDF network architecture
├── utils/
│   ├── edit.py                # Semantic editing & inversion
│   ├── preprocess_sdf.py      # SDF data preprocessing
│   ├── decomposition.py       # Mesh segmentation (optional)
│   └── download_chairs.py     # Dataset download helper
├── data/                      # Training data (gitignored)
├── END_EVALUATION_REPORT.docx # Project documentation
└── README.md
```

## How It Works

### 1. DeepSDF Architecture
- 8-layer MLP with skip connections
- Input: (x, y, z, latent_code)
- Output: signed distance value
- Trained to predict SDF = 0 at surfaces

### 2. Semantic Direction Discovery
- Sample shapes from latent space
- Measure geometric attributes (bbox dimensions)
- Linear regression to find attribute directions
- Result: Interpretable editing controls

### 3. Mesh Reconstruction
- Marching cubes on 128³ SDF grid
- Component filtering (remove fragments)
- Laplacian smoothing (1 iteration)
- Export as GLB format

## Performance

- **Generation Time**: 10-20 seconds per mesh
- **Resolution**: 128³ (2M voxels)
- **Quality**: ~90% of maximum (256³) quality
- **Model Size**: 23MB (hidden_dim=512)

## Advanced Usage

### Environment Variables

For downloading ShapeNet data:
```bash
export HF_TOKEN="your_huggingface_token"
```

### Custom Mesh Upload

1. Prepare mesh in `.glb` or `.obj` format
2. Upload via web interface
3. Wait for latent inversion (100 iterations)
4. Edit using semantic sliders

**Note**: Uploaded meshes are normalized to unit sphere automatically.

## Troubleshooting

### "Reconstruction returned None"
- SDF doesn't cross zero (shape too far from training distribution)
- Solution: Adjust editing magnitude or use different base shape

### Slow generation
- Lower resolution in `app.py`: `create_mesh(..., resolution=64)`
- Trade-off: 4x faster, slightly blockier meshes

### Out of memory
- Reduce batch size in `evaluate.py`: `max_batch=64**3`
- Use CPU instead of GPU for inference

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{editable3dshapes2026,
  title={Editable 3D Shape Priors with Semantic Controls},
  author={Divyansh Garg},
  year={2026},
  howpublished={\url{https://github.com/Yasterd26/editable-3d-shapes}}
}
```

## Acknowledgments

- DeepSDF paper: [Park et al., 2019](https://arxiv.org/abs/1901.05103)
- ShapeNet dataset
- Gradio framework

## License

MIT License - see LICENSE file for details

## Contact

For questions or issues, please open a GitHub issue.
