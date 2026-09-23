# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-FL project

cmake_minimum_required(VERSION 3.26)
project(vllm_fl_extensions LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED VLLM_VENDOR)
    if(DEFINED ENV{VLLM_VENDOR})
        set(VLLM_VENDOR $ENV{VLLM_VENDOR})
    endif()
endif()

if(NOT VLLM_VENDOR)
    message(FATAL_ERROR
        "VLLM_VENDOR is required but not specified.\n"
        "Set VLLM_VENDOR=cuda or ascend."
    )
endif()

set(SUPPORTED_VENDORS cuda ascend)
if(NOT VLLM_VENDOR IN_LIST SUPPORTED_VENDORS)
    message(FATAL_ERROR
        "Unsupported vendor: ${VLLM_VENDOR}\n"
        "Supported vendors: ${SUPPORTED_VENDORS}"
    )
endif()

message(STATUS "vLLM-FL Extensions vendor: ${VLLM_VENDOR}")

if(VLLM_PYTHON_EXECUTABLE)
    set(Python_EXECUTABLE ${VLLM_PYTHON_EXECUTABLE})
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
message(STATUS "Python: ${Python_EXECUTABLE} (${Python_VERSION})")

execute_process(
    # Query package metadata without initializing an installed accelerator
    # backend. Ascend wheel builds intentionally run without a mounted NPU or
    # host driver, while torch 2.10 otherwise auto-loads torch_npu during
    # ``import torch`` and fails on libascend_hal.so before CMake can locate
    # TorchConfig.cmake.
    COMMAND ${CMAKE_COMMAND} -E env TORCH_DEVICE_BACKEND_AUTOLOAD=0
            ${Python_EXECUTABLE} -c
            "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE TORCH_CMAKE_PREFIX_ERROR
    RESULT_VARIABLE TORCH_CMAKE_PREFIX_RESULT
)
if(NOT TORCH_CMAKE_PREFIX_RESULT EQUAL 0 OR NOT TORCH_CMAKE_PREFIX)
    message(FATAL_ERROR
        "Failed to query torch CMake prefix with accelerator autoload disabled: "
        "${TORCH_CMAKE_PREFIX_ERROR}"
    )
endif()
list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX})

find_package(Torch REQUIRED)
message(STATUS "PyTorch: ${Torch_VERSION}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(${VLLM_VENDOR})
