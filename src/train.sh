#!/bin/bash

VENV="/data/rmansouri1/code/sun-sim/venv"

source "$VENV/bin/activate"

echo "Python:   $(which python)"
echo "PyTorch:  $(python -c 'import torch; print(torch.__version__)')"
echo "Python exe: $(python -c 'import sys; print(sys.executable)')"

get_free_port() {
    python -c "
import socket
s = socket.socket()
s.bind(('', 0))
port = s.getsockname()[1]
s.close()
print(port)
"
}

PORT=$(get_free_port)
echo "Using master port: $PORT"

python -m torch.distributed.run \
    --nproc_per_node=2 \
    --master_port="$PORT" \
    train.py "$@"
