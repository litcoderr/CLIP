container_name=$1

cd ..
CUDA_VISIBLE_DEVICES='7'

docker run --gpus '"'device=$CUDA_VISIBLE_DEVICES'"' --ipc=host --rm -it \
    --name $container_name \
    --mount src=$(pwd),dst=/CLIP,type=bind \
    --mount src=/media/data2/,dst=/data,type=bind \
    -e NVIDIA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    -w /CLIP \
    litcoderr/clip_laplacian:latest \
    bash -c "bash" \
