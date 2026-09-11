# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.26)
project(vllm_fl_ascend LANGUAGES CXX)

# The setuptools build uses this directory in two modes. Normal mode builds
# the small torch dispatcher bridge. build_opp.sh configures the exact same
# current-vllm-ascend-0.24.0rc1 source closure as an OPP package.
if(FL_BUILD_CANN_OPP OR "$ENV{FL_BUILD_CANN_OPP}" STREQUAL "1")
  include(${CMAKE_CURRENT_LIST_DIR}/opp_project.cmake)
  return()
endif()

if(NOT ASCEND_HOME_PATH)
  set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
endif()
if(NOT ASCEND_HOME_PATH)
  set(ASCEND_HOME_PATH "/usr/local/Ascend/ascend-toolkit/latest")
endif()

if(NOT TORCH_NPU_PATH)
  execute_process(
    COMMAND ${Python_EXECUTABLE} -c
            "import importlib.util,pathlib; s=importlib.util.find_spec('torch_npu'); print(pathlib.Path(next(iter(s.submodule_search_locations))).resolve())"
    OUTPUT_VARIABLE TORCH_NPU_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TORCH_NPU_QUERY_RESULT
  )
  if(NOT TORCH_NPU_QUERY_RESULT EQUAL 0)
    message(FATAL_ERROR "torch_npu is required for VLLM_VENDOR=ascend")
  endif()
endif()

add_library(_C_ascend SHARED
  torch_binding.cpp
  aclnn_torch_adapter/NPUBridge.cpp
  aclnn_torch_adapter/NPUStorageImpl.cpp
)
target_include_directories(_C_ascend PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${TORCH_INCLUDE_DIRS}
  ${TORCH_NPU_PATH}/include
  ${ASCEND_HOME_PATH}/include
)
target_link_directories(_C_ascend PRIVATE
  ${TORCH_LIBRARY_DIRS}
  ${TORCH_NPU_PATH}/lib
  ${ASCEND_HOME_PATH}/lib64
)
target_link_libraries(_C_ascend PRIVATE
  ${TORCH_LIBRARIES}
  torch_npu
  ascendcl
  dl
  opapi
)
set_target_properties(_C_ascend PROPERTIES
  PREFIX ""
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/ascend"
  BUILD_RPATH "$ORIGIN/_cann_ops_custom/vendors/custom_transformer/op_api/lib"
  INSTALL_RPATH "$ORIGIN/_cann_ops_custom/vendors/custom_transformer/op_api/lib"
)
