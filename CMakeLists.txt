cmake_minimum_required(VERSION 3.24)

# Point CMake at the CUDA toolkit if it is not on PATH, e.g.
#   cmake -B build -DCMAKE_CUDA_COMPILER=$HOME/cuda/bin/nvcc
project(hpc_mri_denoise LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# RTX 4050 Laptop = Ada (sm_89). Override with -DCMAKE_CUDA_ARCHITECTURES=...
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_library(denoise_core
  src/volume_io.cpp
  src/cpu/bilateral.cpp
  src/cpu/nlm.cpp
  src/cpu/diffusion.cpp
  src/cpu/segment.cpp
  src/cuda/bilateral.cu
  src/cuda/nlm.cu
  src/cuda/diffusion.cu
  src/cuda/segment.cu
  src/cuda/device.cu
)
target_include_directories(denoise_core PUBLIC include)
target_link_libraries(denoise_core PUBLIC OpenMP::OpenMP_CXX CUDA::cudart)
target_compile_options(denoise_core PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo --use_fast_math>
)

add_executable(denoise src/main.cpp)
target_link_libraries(denoise PRIVATE denoise_core)
target_compile_options(denoise PRIVATE -O3 -Wall -Wextra)

add_executable(segment src/segment_main.cpp)
target_link_libraries(segment PRIVATE denoise_core)
target_compile_options(segment PRIVATE -O3 -Wall -Wextra)
