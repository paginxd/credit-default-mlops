@echo off
echo Building Docker image...
docker build -t credit-default-api .
if errorlevel 1 (
    echo Docker build failed!
    pause
    exit /b 1
)

echo Starting API container...
docker run --rm -p 8000:8000 credit-default-api
