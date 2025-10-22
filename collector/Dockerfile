FROM nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc0.post1

WORKDIR /workspace

RUN git clone https://github.com/NVIDIA/nccl-tests.git \
      & cd nccl-tests \
      & make MPI=1 MPI_HOME=/usr/local/mpi -j
ENV PATH=/workspace/nccl-tests/build:$PATH

RUN pip3 install --no-cache-dir zeus
