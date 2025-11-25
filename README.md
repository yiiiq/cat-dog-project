# 🐱🐶 Cat vs Dog Image Classifier

A complete end-to-end machine learning project that classifies images as cats or dogs, featuring cloud-based training, REST API deployment, and an interactive web interface.

## 📋 Project Overview and Goals

This project demonstrates a full ML pipeline including:
- Training a CNN model on image data stored in Google Cloud Storage
- Deploying a production-ready REST API using FastAPI
- Creating an interactive web interface with Streamlit
- Containerizing the application with Docker
- Deploying to cloud platforms (Google Cloud Run and Streamlit Cloud)

**Goals:**
- Build a reliable binary image classifier with decent accuracy
- Create a publicly accessible API for image classification
- Provide an easy-to-use web interface for non-technical users
- Demonstrate MLOps best practices with reproducible deployments

## 📊 Dataset Description

**Dataset:** [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data) from Kaggle

**Details:**
- **Total Images:** 25,000 labeled images (12,500 cats, 12,500 dogs)
- **Training Set:** 2,000 images (1,000 cats, 1,000 dogs) - randomly sampled
- **Validation Set:** 20% of training data (400 images)
- **Image Size:** Resized to 128x128 pixels
- **Storage:** Images stored in Google Cloud Storage bucket `image-binary-dataset`
  - Path: `gs://image-binary-dataset/processed/cats/` and `gs://image-binary-dataset/processed/dogs/`
- **Preprocessing:** Images are resized, normalized, and augmented with random flips, rotations, and color jitter

**Data Pipeline:**
1. Raw images uploaded to GCS using `src/preprocess_and_upload.py`
2. Training script loads images directly from cloud storage
3. Real-time data augmentation during training

## 🧠 Model Architecture and Evaluation

### Architecture: Simple CNN

```
Input (3x128x128 RGB image)
    ↓
Conv2D (32 filters, 3x3) → ReLU → MaxPool (2x2)
    ↓
Conv2D (64 filters, 3x3) → ReLU → MaxPool (2x2)
    ↓
Conv2D (128 filters, 3x3) → ReLU → MaxPool (2x2)
    ↓
Flatten → FC (512) → ReLU → Dropout (0.5)
    ↓
FC (1) → Sigmoid
    ↓
Output (probability: 0=cat, 1=dog)
```

**Model Details:**
- **Parameters:** ~16.8 million
- **Loss Function:** Binary Cross-Entropy
- **Optimizer:** Adam (learning rate: 0.001)
- **Training:** 5 epochs, batch size 32

### Evaluation Metrics

**Validation Performance:**
- **Accuracy:** ~90%
- **Precision:** ~0.89
- **Recall:** ~0.91
- **F1-Score:** ~0.90

**Experiment Tracking:**
- MLflow used for logging metrics, parameters, and model artifacts
- Local tracking in `./mlruns` directory

## ☁️ Cloud Services Used

### Google Cloud Platform (GCP)

1. **Google Cloud Storage (GCS)**
   - Bucket: `image-binary-dataset`
   - Stores preprocessed training images
   - Cost: ~$0.10/month

2. **Google Cloud Run**
   - Hosts the FastAPI prediction service
   - Serverless container deployment
   - Auto-scaling from 0 to N instances
   - Configuration: 2 vCPU, 2GB RAM, 300s timeout
   - Cost: Free tier covers typical usage (2M requests/month)

3. **Google Container Registry (GCR)**
   - Stores Docker images
   - Integrated with Cloud Build

4. **Google Cloud Build**
   - Automated Docker image building
   - CI/CD integration

### Streamlit Cloud

- **Hosting:** Free tier for public apps
- **Purpose:** Interactive web UI for image uploads
- **Integration:** Calls Cloud Run API for predictions

## 🚀 Setup and Usage Instructions

### Prerequisites

- Python 3.9+
- Docker
- Google Cloud SDK (`gcloud`)
- Git

### Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yiiiq/cat-dog-project.git
   cd cat-dog-project
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Authenticate with Google Cloud:**
   ```bash
   gcloud auth application-default login
   ```

4. **Upload training data to GCS (optional - if you have raw data):**
   ```bash
   python src/preprocess_and_upload.py
   ```

5. **Train the model:**
   ```bash
   python src/models/train.py
   ```
   Model will be saved to `models/best_model.pth`

6. **Run the API locally:**
   ```bash
   uvicorn src.backend.api:app --reload
   # Access at http://localhost:8000
   # API docs at http://localhost:8000/docs
   ```

7. **Run the web interface locally:**
   ```bash
   cd src/ui
   streamlit run app.py
   # Opens at http://localhost:8501
   ```

### Docker Usage

**Build and run the containerized API:**

```bash
# Build image
docker build -t cat-dog-classifier .

# Run container
docker run -p 8000:8000 -v $(pwd)/models:/app/models cat-dog-classifier
```

### Cloud Deployment

**Deploy API to Google Cloud Run:**

```bash
# Set your project ID
export GCP_PROJECT_ID="your-gcp-project-id"

# Deploy
./deploy_to_cloudrun.sh
```

**Deploy UI to Streamlit Cloud:**

1. Push code to GitHub
2. Go to https://share.streamlit.io/
3. Deploy `src/ui/app.py` from your repository
4. Add API URL to Streamlit secrets

See [STREAMLIT_DEPLOYMENT.md](STREAMLIT_DEPLOYMENT.md) for detailed instructions.

## 🌐 Deployed Applications

### API Endpoint
**URL:** `https://cat-dog-classifier-wvwq66fufa-uc.a.run.app`

**Available Endpoints:**
- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Upload image for classification
- `GET /docs` - Interactive API documentation (Swagger UI)

**Example API Usage:**
```bash
# Health check
curl https://cat-dog-classifier-wvwq66fufa-uc.a.run.app/health

# Predict
curl -X POST "https://cat-dog-classifier-wvwq66fufa-uc.a.run.app/predict" \
  -F "file=@image.jpg"
```

**Response Format:**
```json
{
  "prediction": "cat",
  "confidence": 0.92,
  "probabilities": {
    "cat": 0.92,
    "dog": 0.08
  }
}
```

### Web Interface
**URL:** `https://[your-username]-cat-dog-project-[hash].streamlit.app`

**Features:**
- Drag-and-drop image upload
- Real-time predictions
- Confidence scores and probability visualization
- Mobile-friendly responsive design

## 📁 Project Structure

```
cat-dog-project/
├── src/
│   ├── models/
│   │   └── train.py              # Model training script
│   ├── backend/
│   │   └── api.py                # FastAPI application
│   ├── ui/
│   │   ├── app.py                # Streamlit web interface
│   │   └── requirements.txt      # UI dependencies
│   └── preprocess_and_upload.py  # Data preprocessing
├── models/
│   └── best_model.pth            # Trained model weights
├── configs/
│   └── config.yaml               # Training configuration
├── Dockerfile                    # Container configuration
├── requirements.txt              # Python dependencies
├── deploy_to_cloudrun.sh        # Deployment script
└── README.md                     # This file
```

## 🔧 Configuration

Training parameters in `configs/config.yaml`:

```yaml
data:
  cloud_storage: "gcs"
  bucket: "image-binary-dataset"
  processed_path: "processed"

train:
  model: "SimpleCNN"
  epochs: 5
  batch_size: 32
  lr: 0.001
  val_split: 0.2
  seed: 42
```


## 🙏 Acknowledgments

- Dataset: [Kaggle Dogs vs Cats Competition](https://www.kaggle.com/c/dogs-vs-cats)
- Frameworks: PyTorch, FastAPI, Streamlit
- Cloud Platform: Google Cloud Platform
