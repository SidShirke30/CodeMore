param(
    [string]$ImageName = "week15-model-api",
    [string]$ContainerName = "week15-model-api",
    [int]$Port = 5000
)

Write-Host "Building Docker image..."
docker build -t $ImageName .

if ($LASTEXITCODE -ne 0) {
    throw "Docker build failed."
}

Write-Host "Removing previous container if it exists..."
docker rm -f $ContainerName 2>$null

Write-Host "Starting deployment..."
docker run -d `
    --name $ContainerName `
    -p "$Port`:5000" `
    $ImageName

if ($LASTEXITCODE -ne 0) {
    throw "Container deployment failed."
}

Write-Host "Deployment started successfully."
Write-Host "Health endpoint: http://localhost:$Port/health"
