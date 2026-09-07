source scripts/corex_env.sh

export DETECTOR_DEVICE=0
export DETECTOR_GRPC_HOST=0.0.0.0
export DETECTOR_GRPC_PORT=50051
export DETECTOR_AUTH_TOKEN='change-this-to-a-long-random-secret'

export DETECTOR_WEIGHTS_SHA256='8b3b748c1e592ddd8868022e8732fde20025197328490623cc16c6f24d0782ee'

scripts/run_gpu_detector.sh