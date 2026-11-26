# Dockerfile for TTM-R2 Inference Service
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including git
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY ttm_requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download TTM-R2 model (optional - can be done at runtime)
# RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('ibm-granite/granite-timeseries-ttm-r2', trust_remote_code=True)"

# Copy proto file
COPY proto/ ./proto/

# Generate gRPC code
RUN python -m grpc_tools.protoc \
    -I./proto \
    --python_out=. \
    --grpc_python_out=. \
    ./proto/ttm_service.proto

# Copy service code
COPY server.py model_loader.py anomaly_detector.py preprocessor.py ./

# Expose gRPC port
EXPOSE 50051

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import grpc; from ttm_service_pb2_grpc import TTMServiceStub; from ttm_service_pb2 import HealthRequest; channel = grpc.insecure_channel('localhost:50051'); stub = TTMServiceStub(channel); stub.HealthCheck(HealthRequest())"

# Run server
CMD ["python", "server.py", "--port", "50051"]