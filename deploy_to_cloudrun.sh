#!/bin/bash

# Google Cloud Run Deployment Script
# This deploys the cat-dog classifier API to Google Cloud Run

set -e  # Exit on error

echo "======================================"
echo "Deploying to Google Cloud Run"
echo "======================================"

# Configuration - EDIT THESE VALUES
PROJECT_ID="${GCP_PROJECT_ID:-your-gcp-project-id}"
SERVICE_NAME="cat-dog-classifier"
REGION="us-central1"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

# Check if PROJECT_ID is set
if [ "$PROJECT_ID" = "your-gcp-project-id" ]; then
    echo "Error: Please set GCP_PROJECT_ID environment variable"
    echo "Example: export GCP_PROJECT_ID=my-project-123"
    exit 1
fi

echo "Project ID: $PROJECT_ID"
echo "Service Name: $SERVICE_NAME"
echo "Region: $REGION"
echo ""

# Step 1: Set the project
echo "[1/5] Setting GCP project..."
gcloud config set project $PROJECT_ID

# Step 2: Enable required APIs
echo ""
echo "[2/5] Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Step 3: Build and push the container
echo ""
echo "[3/5] Building and pushing container image..."
echo "This will take 10-20 minutes..."

# Use Cloud Build to build the image (faster than local build + push)
gcloud builds submit --tag ${IMAGE_NAME} .

# Step 4: Deploy to Cloud Run
echo ""
echo "[4/5] Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
  --image ${IMAGE_NAME} \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --port 8000 \
  --command uvicorn \
  --args src.backend.api:app,--host,0.0.0.0,--port,8000

# Step 5: Get the service URL
echo ""
echo "[5/5] Getting service URL..."
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
  --region ${REGION} \
  --format 'value(status.url)')

echo ""
echo "======================================"
echo "✓ Deployment Complete!"
echo "======================================"
echo ""
echo "Service URL: $SERVICE_URL"
echo ""
echo "Test your API:"
echo "  Health check:"
echo "    curl ${SERVICE_URL}/health"
echo ""
echo "  Prediction (replace with your image path):"
echo "    curl -X POST \"${SERVICE_URL}/predict\" \\"
echo "      -F \"file=@data/raw/train/cat.1.jpg\""
echo ""
echo "Interactive docs: ${SERVICE_URL}/docs"
echo ""
