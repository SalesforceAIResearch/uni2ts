#!/bin/bash
# Script to build and start the Python sandbox Docker container in background
# and expose the HTTP API interface

set -e

IMAGE_NAME="python-sandbox"
CONTAINER_NAME="python-sandbox-server"
PORT=${SANDBOX_PORT:-8080}

echo "=========================================="
echo "Python Sandbox - Starting Background Server"
echo "=========================================="

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Build the Docker image if it doesn't exist or if --rebuild is passed
if [ "$1" == "--rebuild" ] || ! docker image inspect $IMAGE_NAME > /dev/null 2>&1; then
    echo ""
    echo "Building Docker image..."
    docker build -t $IMAGE_NAME "$SCRIPT_DIR"
else
    echo ""
    echo "Docker image already exists, skipping build..."
    echo "Use '$0 --rebuild' to rebuild the image"
fi

# Check if container already exists and remove it
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo ""
    echo "Stopping and removing existing container..."
    docker stop $CONTAINER_NAME > /dev/null 2>&1 || true
    docker rm $CONTAINER_NAME > /dev/null 2>&1 || true
fi

# Start the container in background
echo ""
echo "Starting container in background..."
echo "Container name: $CONTAINER_NAME"
echo "Exposed port: $PORT"
echo ""

docker run -d \
    --name $CONTAINER_NAME \
    -p $PORT:8080 \
    -e SANDBOX_PORT=8080 \
    $IMAGE_NAME

# Wait a moment for the server to start
sleep 2

# Check if container is running
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "✅ Container is running!"
    echo ""
    echo "Server is available at: http://localhost:$PORT"
    echo "Health check: http://localhost:$PORT/health"
    echo "Execute endpoint: http://localhost:$PORT/execute"
    echo ""
    echo "To view logs: docker logs -f $CONTAINER_NAME"
    echo "To stop: docker stop $CONTAINER_NAME"
    echo "To remove: docker rm $CONTAINER_NAME"
    echo ""
    
    # Test health endpoint
    echo "Testing health endpoint..."
    if curl -s http://localhost:$PORT/health > /dev/null; then
        echo "✅ Health check passed!"
    else
        echo "⚠️  Health check failed, but container is running. Check logs with: docker logs $CONTAINER_NAME"
    fi
else
    echo "❌ Failed to start container. Check logs with: docker logs $CONTAINER_NAME"
    exit 1
fi

echo "=========================================="

