# GLoop: An Event-driven Runtime for Consolidating GPGPU Applications

## Requirements

- GCC 4.9 (You need to modify /usr/lib/nvidia-cuda-toolkit/{gcc, g++})
- NVIDIA CUDA 7.5 or later
- grpc
- CMake with CUDA patch (https://github.com/CPFL/cmake cuda branch)

## Development

Do not use `master` branch, it is highly focusing on development purpose.

We have branches, `kepler` for Kepler K40c and `pascal` for Pascal P100.

## Environment

- Ubuntu 16.04

## Publications

- Yusuke Suzuki, Hiroshi Yamada, Shinpei Kato and Kenji Kono: **GLoop: An Event-driven Runtime for Consolidating GPGPU Applications**, In Proceedings of the 8th ACM Symposium on Cloud Computing (SoCC '17), 2017.
- Yusuke Suzuki, Hiroshi Yamada, Shinpei Kato and Kenji Kono: __Towards Multi-tenant GPGPU: Event-driven Programming Model for System-wide Scheduling on Shared GPUs__, In The 2017 Workshop on Multicore and Rack-scale Systems (MaRS '16), 2016.
