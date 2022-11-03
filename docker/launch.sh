cd ..
CUDA_VISIBLE_DEVICES='2,5'

docker run --gpus '"'device=$CUDA_VISIBLE_DEVICES'"' --ipc=host --rm -it \
    --mount src=$(pwd),dst=/CLIP,type=bind \
    --mount src=/media/user/data2,dst=/data,type=bind \
    -e NVIDIA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    -w /CLIP \
    litcoderr/clip_laplacian:latest \
    bash -c "bash" \
