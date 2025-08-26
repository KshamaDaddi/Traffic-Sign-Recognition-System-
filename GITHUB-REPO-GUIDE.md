# Traffic Sign Recognition System - Complete Implementation

This repository contains a complete implementation of an intelligent traffic sign recognition system using deep learning and computer vision techniques.

## 📁 Repository Structure

```
traffic-sign-recognition/
├── 📄 Core Implementation
│   ├── traffic_sign_model.py      # CNN architectures (Custom, InceptionV3, VGG16, ResNet50)
│   ├── data_utils.py              # Data processing and augmentation utilities
│   ├── train.py                   # Training script with multiple architecture support
│   ├── evaluate.py                # Model evaluation and testing
│   └── realtime_detection.py      # Real-time webcam/video detection
│
├── 🌐 Web Deployment
│   ├── streamlit_app.py           # Interactive web application
│   └── fastapi_app.py             # REST API service
│
├── 🛠️ Utilities & Analysis
│   ├── test_installation.py       # Installation verification
│   ├── compare_models.py          # Model performance comparison
│   └── dataset_info.py            # Dataset analysis and visualization
│
├── ⚙️ Configuration & Setup
│   ├── requirements.txt           # Python dependencies
│   ├── Dockerfile                 # Container configuration
│   ├── setup.sh                   # Linux/Mac setup script
│   ├── setup.bat                  # Windows setup script
│   └── README.md                  # Complete documentation
│
└── 📂 Directories
    ├── data/                      # Dataset storage
    ├── outputs/                   # Trained models and results
    ├── models/                    # Model architectures
    └── evaluation_results/        # Evaluation outputs
```

## 🚀 Quick Start Guide

### 1. Installation & Setup

**Option A: Automated Setup (Recommended)**
```bash
# Clone repository
git clone https://github.com/yourusername/traffic-sign-recognition.git
cd traffic-sign-recognition

# Run setup (Linux/Mac)
chmod +x setup.sh && ./setup.sh

# Run setup (Windows)
setup.bat
```

**Option B: Manual Setup**
```bash
# Create virtual environment
python -m venv traffic_sign_env
source traffic_sign_env/bin/activate  # Linux/Mac
# traffic_sign_env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p data outputs models evaluation_results
```

### 2. Verify Installation
```bash
python test_installation.py
```

### 3. Download & Prepare Data
```bash
# Download GTSRB dataset (automatic)
python train.py --download_data

# Analyze dataset
python dataset_info.py
```

### 4. Train Models
```bash
# Train custom CNN (recommended for beginners)
python train.py --architecture custom --epochs 50

# Train with transfer learning
python train.py --architecture InceptionV3 --epochs 30

# Train multiple architectures for comparison
python train.py --architecture VGG16 --epochs 40
python train.py --architecture ResNet50 --epochs 35
```

### 5. Compare Model Performance
```bash
python compare_models.py
```

### 6. Evaluate Models
```bash
# Evaluate on test set
python evaluate.py --model_path outputs/final_model_custom.h5

# Test on your own images
python evaluate.py --model_path outputs/final_model_custom.h5 --test_images image1.jpg image2.jpg
```

### 7. Real-time Detection
```bash
# Webcam detection
python realtime_detection.py --model_path outputs/final_model_custom.h5 --mode webcam

# Process video file
python realtime_detection.py --model_path outputs/final_model_custom.h5 --mode video --video_path input.mp4
```

### 8. Deploy Web Applications
```bash
# Streamlit web app
streamlit run streamlit_app.py

# FastAPI service
python fastapi_app.py
```

## 🎯 Training Options & Parameters

### Model Architectures
- **`custom`**: Optimized CNN for traffic signs (fastest training, good accuracy)
- **`InceptionV3`**: Google's Inception v3 (best for transfer learning)
- **`VGG16`**: Visual Geometry Group 16-layer (stable, reliable)
- **`ResNet50`**: Residual Network 50-layer (good for complex features)

### Training Commands
```bash
# Basic training
python train.py --architecture custom

# Advanced training with all options
python train.py \
    --architecture InceptionV3 \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --augment_data \
    --data_path data/ \
    --output_dir outputs/

# Quick training (for testing)
python train.py --architecture custom --epochs 10 --batch_size 64
```

## 📊 Performance Benchmarks

### Expected Results (GTSRB Dataset)
| Architecture | Accuracy | Training Time | Model Size | Inference Speed |
|--------------|----------|---------------|------------|-----------------|
| Custom CNN   | 98.2%    | ~2 hours      | 15 MB      | 5ms            |
| InceptionV3  | 97.1%    | ~3 hours      | 45 MB      | 8ms            |
| VGG16        | 98.0%    | ~4 hours      | 35 MB      | 12ms           |
| ResNet50     | 96.5%    | ~3.5 hours    | 25 MB      | 10ms           |

*Results may vary based on hardware and training parameters*

## 🔧 Advanced Usage

### Custom Dataset Training
```python
from data_utils import TrafficSignDataProcessor
from traffic_sign_model import TrafficSignCNN

# Initialize processor for custom data
processor = TrafficSignDataProcessor(data_path='my_custom_data/')

# Load your data (implement your loading logic)
X_train, y_train, X_test, y_test = load_my_custom_data()

# Preprocess
X_train, y_train, X_val, y_val, X_test, y_test = processor.preprocess_data(
    X_train, y_train, X_test, y_test
)

# Train model
model = TrafficSignCNN(num_classes=len(my_classes))
model.compile_model()
history = model.train(X_train, y_train, X_val, y_val)
```

### Hyperparameter Tuning
```bash
# Grid search example
for lr in 0.001 0.0005 0.0001; do
    for bs in 16 32 64; do
        python train.py --architecture custom --learning_rate $lr --batch_size $bs --epochs 20
    done
done

# Then compare results
python compare_models.py
```

### Model Ensemble
```python
from tensorflow.keras.models import load_model
import numpy as np

# Load multiple models
models = [
    load_model('outputs/final_model_custom.h5'),
    load_model('outputs/final_model_InceptionV3.h5'),
    load_model('outputs/final_model_VGG16.h5')
]

# Ensemble prediction
def ensemble_predict(models, X):
    predictions = [model.predict(X) for model in models]
    return np.mean(predictions, axis=0)
```

## 🌐 Deployment Options

### 1. Local Streamlit App
```bash
pip install streamlit plotly
streamlit run streamlit_app.py
# Access: http://localhost:8501
```

### 2. FastAPI REST Service
```bash
pip install fastapi uvicorn
python fastapi_app.py
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

### 3. Docker Deployment
```bash
# Build image
docker build -t traffic-sign-recognition .

# Run container
docker run -p 8000:8000 traffic-sign-recognition

# With GPU support
docker run --gpus all -p 8000:8000 traffic-sign-recognition
```

### 4. Cloud Deployment

**AWS Lambda**
```bash
# Install serverless framework
npm install -g serverless

# Package for Lambda
pip install -t . tensorflow-lite-runtime
zip -r lambda-deployment.zip .
```

**Google Cloud Run**
```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT-ID/traffic-sign-recognition
gcloud run deploy --image gcr.io/PROJECT-ID/traffic-sign-recognition --platform managed
```

## 🛠️ Troubleshooting

### Common Issues & Solutions

**1. CUDA/GPU Issues**
```bash
# Check CUDA
nvidia-smi

# Install GPU TensorFlow
pip install tensorflow[and-cuda]==2.13.0

# If still issues, use CPU version
pip install tensorflow-cpu==2.13.0
```

**2. Memory Errors During Training**
```bash
# Reduce batch size
python train.py --batch_size 16

# Enable memory growth
export TF_FORCE_GPU_ALLOW_GROWTH=true
```

**3. Dataset Download Issues**
```bash
# Manual download if automatic fails
cd data/
wget https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip
unzip GTSRB_Final_Training_Images.zip
```

**4. Model Loading Errors**
```python
# Load with custom objects
from tensorflow.keras.models import load_model
model = load_model('model.h5', compile=False)
```

### Performance Optimization

**Training Speed**
- Use GPU: `pip install tensorflow[and-cuda]`
- Increase batch size: `--batch_size 64`
- Mixed precision: `export TF_ENABLE_MIXED_PRECISION_GRAPH_REWRITE=1`

**Inference Speed**
- Model quantization for mobile deployment
- TensorRT optimization for NVIDIA GPUs
- ONNX conversion for cross-platform deployment

## 📈 Monitoring & Analytics

### Training Progress
```bash
# Monitor with TensorBoard
pip install tensorboard
tensorboard --logdir outputs/logs
```

### Model Performance Tracking
```bash
# Generate detailed reports
python evaluate.py --model_path outputs/final_model_custom.h5 > evaluation_report.txt

# Compare multiple models
python compare_models.py
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature-amazing-feature`  
5. Open Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black *.py
flake8 *.py
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **GTSRB Dataset**: Institut für Neuroinformatik - Ruhr-Universität Bochum
- **TensorFlow Team**: For the excellent deep learning framework
- **OpenCV Community**: For computer vision tools
- **Streamlit & FastAPI**: For web deployment frameworks

## 📞 Support & Community

- **Issues**: [GitHub Issues](https://github.com/yourusername/traffic-sign-recognition/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/traffic-sign-recognition/discussions)
- **Email**: support@traffic-sign-recognition.com

---

**🚦 Built for Road Safety & Autonomous Driving Research 🚗**