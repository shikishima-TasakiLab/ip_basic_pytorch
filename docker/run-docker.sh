#!/bin/bash
RUN_DIR=$(dirname $(readlink -f $0))

DATASET_DIR='/data/hdf5'

DOCKER_VOLUME="${DOCKER_VOLUME} -v $(dirname ${RUN_DIR}):/workspace/ip_basic:rw"
DOCKER_VOLUME="${DOCKER_VOLUME} -v ${DATASET_DIR}:/workspace/dataset:ro"

docker run \
    -it \
    --rm \
    --gpus '"device=0"' \
    ${DOCKER_VOLUME} \
    --name IP-Basic \
    g3od
