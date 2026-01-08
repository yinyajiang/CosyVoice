#!/bin/bash

set -euo pipefail
cd $(dirname $0)

show_help() {
    echo "Usage: $0 --scope <scope>"
    echo ""
    echo "Options:"
    echo "  --scope <scope>    Specify build scope, can be: env, model, code"
    echo "                     Multiple scopes can be specified separated by comma"
    echo "                     Example: --scope env,model,code"
    echo ""
    echo "Build order: env -> model -> code"
    exit 1
}

SCOPE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --scope)
            SCOPE="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# 如果没有指定scope，报错并打印help
if [ -z "$SCOPE" ]; then
    echo "Error: --scope parameter is required"
    echo ""
    show_help
fi

# 按照顺序构建：env -> model -> code
if [[ "$SCOPE" == *"env"* ]]; then
    echo "Building Dockerfile_env..."
    docker build -t cosyvoice_env:latest -f ./docker/Dockerfile_env .
fi

if [[ "$SCOPE" == *"model"* ]]; then
    echo "Building Dockerfile_model..."
    docker build -t cosyvoice_model:latest -f ./docker/Dockerfile_model .
fi

if [[ "$SCOPE" == *"code"* ]]; then
    echo "Building Dockerfile_code..."
    docker build -t cosyvoice_code:latest -f ./docker/Dockerfile_code .
fi