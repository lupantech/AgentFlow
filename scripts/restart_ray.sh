#!/bin/bash

set -ex

ray stop -v --force --grace-period 60
ps aux
env RAY_DEBUG=legacy HYDRA_FULL_ERROR=1 VLLM_USE_V1=1 CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" ray start --head --dashboard-host=0.0.0.0
