#!/bin/bash

set -euo pipefail
cd $(dirname $0)

MODEL="pretrained_models/Fun-CosyVoice3-0.5B"

python test_trt_memory.py --model "${1:-$MODEL}"



