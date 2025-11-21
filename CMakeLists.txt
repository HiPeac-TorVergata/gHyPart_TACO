cmake_minimum_required(VERSION 3.18)  # modern CUDA arch detection needs ≥ 3.18
project(gpuHgrPar LANGUAGES CXX CUDA)

# -----------------------------
#  Detect CUDA (stop if missing)
# -----------------------------
include(CheckLanguage)
check_language(CUDA)

if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "CUDA compiler not found! Install CUDA or check environment variables.")
endif()

enable_language(CUDA)

# --- Detect GPU compute capability using nvidia-smi ------------------------
execute_process(
    COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
    OUTPUT_VARIABLE GPU_COMPUTE_CAP_RAW
    ERROR_VARIABLE GPU_SMI_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
)

if(GPU_COMPUTE_CAP_RAW MATCHES "^[0-9]+\\.[0-9]+$")
    string(REPLACE "." "" GPU_ARCH "${GPU_COMPUTE_CAP_RAW}")
    message(STATUS "Detected compute capability: ${GPU_COMPUTE_CAP_RAW} → sm_${GPU_ARCH}")
    set(CMAKE_CUDA_ARCHITECTURES ${GPU_ARCH})
else()
    message(WARNING
        "Could not parse compute capability from nvidia-smi.\n"
        "nvidia-smi error: ${GPU_SMI_ERROR}\n"
        "Falling back to sm_86."
    )
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()
# ---------------------------------------------------------------------------


# -----------------------------
#  Compilation flags
# -----------------------------
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -lineinfo -fopenmp")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(
    kernels/
    include/
    utility/
    no_uvm_impl/
    no_uvm_impl/non-deterministic/
    use_uvm_impl/
    .
)

# -----------------------------
#   Sources
# -----------------------------
set(GPU_HGR_PAR_SOURCE_FILES
    kernels/coarsening_kernels.cuh
    kernels/coarsening_kernels.cu
    kernels/construction_kernels.cuh
    kernels/construction_kernels.cu
    kernels/partitioning_kernels.cuh
    kernels/partitioning_kernels.cu
    kernels/refinement_kernels.cuh
    kernels/refinement_kernels.cu
    
    include/partitioning_impl.h
    include/refinement_impl.h
    include/projection.cuh
    include/coarsening_impl.h
    include/construction_impl.h
    include/graph.h

    utility/utils.cuh
    utility/utils.cu
    utility/param_config.h
    utility/param_config.cc
    utility/validation.cuh
    utility/validation.cu

    no_uvm_impl/non-deterministic/nondeterministic_coarsen_no_uvm_kernels.cuh
    no_uvm_impl/non-deterministic/nondeterministic_coarsen_no_uvm_kernels.cu
    no_uvm_impl/non-deterministic/nondeterministic_coarsen_no_uvm.cu

    no_uvm_impl/use_no_uvm.cuh
    no_uvm_impl/coarsen_brute_force.cu
    no_uvm_impl/coarsen_no_uvm_kernels.cuh
    no_uvm_impl/coarsen_no_uvm_kernels_opt.cu
    no_uvm_impl/coarsen_no_uvm_kernels.cu
    no_uvm_impl/coarsen_no_uvm.cu
    no_uvm_impl/partition_no_uvm_kernels.cuh
    no_uvm_impl/partition_no_uvm_kernels.cu
    no_uvm_impl/partition_no_uvm.cu
    no_uvm_impl/refinement_no_uvm_kernels.cuh
    no_uvm_impl/refinement_no_uvm_kernels.cu
    no_uvm_impl/refinement_no_uvm.cu
    no_uvm_impl/use_no_uvm.cu
    no_uvm_impl/kway_partition_no_uvm.cu
    no_uvm_impl/kway_partition_no_uvm_kernels.cuh

    use_uvm_impl/coarsening.cu
    use_uvm_impl/construction.cu
    use_uvm_impl/partitioning_base.cu
    use_uvm_impl/partitioning_opt1.cu
    use_uvm_impl/partitioning_opt2.cu
    use_uvm_impl/refinement_base.cu
    use_uvm_impl/refinement_opt1.cu
    use_uvm_impl/refinement_opt2.cu
    use_uvm_impl/refinement_opt3.cu
    use_uvm_impl/refinement_opt4.cu
    use_uvm_impl/projection.cu

    main.cu
)

# -----------------------------
#   Executable
# -----------------------------
add_executable(gHyPart ${GPU_HGR_PAR_SOURCE_FILES})

find_package(OpenMP REQUIRED)

target_link_libraries(
    gHyPart
    -lstdc++fs
    -lcusparse
    OpenMP::OpenMP_CXX
)
