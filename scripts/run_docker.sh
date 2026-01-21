docker run -it --ipc=host --network=host --group-add render \
        --privileged --security-opt seccomp=unconfined \
        --cap-add=CAP_SYS_ADMIN --cap-add=SYS_PTRACE \
        --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
        -v /remote/vast0/share-mv:/remote/vast0/share-mv \
        -v /remote/vast0/share-mv/tran/workspace:/workspace \
        -w /workspace/VLM_Embed \
        -v /home/tran/.ssh:/root/.ssh \
        --name anhtt-exaone-dev \
        moreh-vllm:anhtt-dev bash
        # rocm/pytorch:rocm7.0.2_ubuntu22.04_py3.10_pytorch_release_2.7.1 bash