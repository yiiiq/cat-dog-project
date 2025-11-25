import os
import io
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms
import uvicorn


# Define the same SimpleCNN model architecture used for training
class SimpleCNN(nn.Module):
    """Simple CNN for binary classification (cat vs dog)"""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling and activation
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers (128x128 input -> 16x16 after 3 pooling)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Conv blocks
        x = self.pool(self.relu(self.conv1(x)))  # 128 -> 64
        x = self.pool(self.relu(self.conv2(x)))  # 64 -> 32
        x = self.pool(self.relu(self.conv3(x)))  # 32 -> 16
        
        # Flatten
        x = x.view(-1, 128 * 16 * 16)
        
        # FC layers
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))
        
        return x


# Initialize FastAPI app
app = FastAPI(
    title="Cat vs Dog Classifier API",
    description="Upload an image to predict whether it's a cat or a dog",
    version="1.0.0"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and device
model = None
device = None
transform = None


def load_model():
    """Load the trained model"""
    global model, device, transform
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = SimpleCNN().to(device)
    
    # Load trained weights
    model_path = 'models/best_model.pth'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded successfully from {model_path}")
    
    # Define image transformation (same as validation transform in training)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


@app.on_event("startup")
async def startup_event():
    """Load model when the application starts"""
    load_model()


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Cat vs Dog Classifier API",
        "endpoints": {
            "/predict": "POST - Upload an image to classify as cat or dog",
            "/health": "GET - Check API health status"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device)
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict whether an uploaded image is a cat or a dog
    
    Args:
        file: Image file (JPEG, PNG, etc.)
    
    Returns:
        JSON with prediction, confidence, and probabilities
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    try:
        # Read and process the image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Apply transformations
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        image_tensor = image_tensor.to(device)
        
        # Make prediction
        with torch.no_grad():
            output = model(image_tensor)
            probability = output.item()
        
        # Interpret results (0 = cat, 1 = dog)
        prediction = "dog" if probability > 0.5 else "cat"
        confidence = probability if probability > 0.5 else 1 - probability
        
        return {
            "prediction": prediction,
            "confidence": float(confidence),
            "probabilities": {
                "cat": float(1 - probability),
                "dog": float(probability)
            }
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
