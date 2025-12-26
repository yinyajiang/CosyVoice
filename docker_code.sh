#!/bin/bash

set -e
cd $(dirname $0)

docker build -t cosyvoice_code -f ./docker/Dockerfile_code  .