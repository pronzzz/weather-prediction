# 1. Set your Google Cloud Project ID
$env:PROJECT_ID = "weather-app-455806"
gcloud config set project $env:PROJECT_ID

# 2. Enable necessary Google Cloud services
gcloud services enable run.googleapis.com `
    artifactregistry.googleapis.com `
    cloudbuild.googleapis.com `
    aiplatform.googleapis.com  # AI Platform may not be strictly needed

# 3. Create an Artifact Registry repository (if you don't have one)
$env:REGION = "us-central1"  # Choose your preferred region
$env:REPOSITORY_NAME = "cloud-deploy-repo"

gcloud artifacts repositories create $env:REPOSITORY_NAME `
    --repository-format=docker `
    --location=$env:REGION `
    --description="Docker repository for cloud deployment apps"

# Verify repository creation (optional)
# gcloud artifacts repositories list --location=$env:REGION

# 4. Define Image Name in Artifact Registry
$env:IMAGE_NAME = "$env:REGION-docker.pkg.dev/$env:PROJECT_ID/$env:REPOSITORY_NAME/weather-classifier:latest"

# 5. Build the Docker image using Google Cloud Build and push it to Artifact Registry
Write-Output "Submitting build to Google Cloud Build..."
gcloud builds submit --tag $env:IMAGE_NAME

# Check if the image exists in Artifact Registry (optional)
gcloud artifacts docker images list "$env:REGION-docker.pkg.dev/$env:PROJECT_ID/$env:REPOSITORY_NAME"

# 6. Deploy the image to Google Cloud Run
$env:SERVICE_NAME = "weather-classifier-service"
Write-Output "Deploying service $env:SERVICE_NAME to Cloud Run in region $env:REGION..."

# Deploy the service to Cloud Run
gcloud run deploy $env:SERVICE_NAME `
    --image=$env:IMAGE_NAME `
    --platform=managed `
    --region=$env:REGION `
    --memory=2Gi `
    --cpu=1 `
    --port=8080 `
    --timeout=300 `
    --concurrency=10 `
    --allow-unauthenticated

# Output deployment status
Write-Output "-------------------------------------------------------"
Write-Output "Deployment submitted. Check the Google Cloud Console for status."
Write-Output "Service URL will be printed upon successful deployment."
Write-Output "-------------------------------------------------------"

# 7. Get the URL of the deployed service
$env:SERVICE_URL = gcloud run services describe $env:SERVICE_NAME `
    --platform=managed `
    --region=$env:REGION `
    --format="value(status.url)"
Write-Output "Service URL: $env:SERVICE_URL"