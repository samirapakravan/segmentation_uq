#!/usr/bin/env bash

IMAGE_NAME="vision"
CONT_NAME=vision

DOCKER_CMD="docker run \
    --network host \
    --gpus all \
    --shm-size=64gb \
    -p ${JUPYTER_PORT}:8888 \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/group:/etc/group:ro \
    -v /etc/shadow:/etc/shadow:ro \
    -v $(pwd):/workspace \
    -e HOME=/workspace \
    -w /workspace"


usage() {
    echo "Please read README."
}


build() {
    set -x
    DOCKER_FILE="docker/Dockerfile"

    echo -e "Building ${DOCKER_FILE}..."
    docker build \
        -t ${IMAGE_NAME}:0.0.1 \
        -t ${IMAGE_NAME}:latest \
        -f ${DOCKER_FILE} .
}


dev() {
    local DEV_IMG=${IMAGE_NAME}

    while [[ $# -gt 0 ]]; do
        case $1 in
            -i|--image)
                DEV_IMG="$2"
                shift
                shift
                ;;
	    -d|--deamon)
                DOCKER_CMD="${DOCKER_CMD} -d"
                shift
                ;;
            *)
                echo "Unknown option $1"
                exit 1
                ;;
        esac
    done

    $DOCKER_CMD \
        --name ${CONT_NAME} \
        -u $(id -u):$(id -u) \
        -e PYTHONPATH=$DEV_PYTHONPATH \
        -it --rm \
        ${DEV_IMG} \
        bash
}


attach() {
    DOCKER_CMD="docker exec"
    CONTAINER_ID=$(docker ps | grep ${CONT_NAME} | cut -d' ' -f1)
    ${DOCKER_CMD} -it ${CONTAINER_ID} /bin/bash
    exit
}

case $1 in
    build)
        "$@"
        ;;
    dev)
        "$@"
        exit 0
        ;;
    attach)
        $@
        ;;
    *)
        usage
        ;;
esac
