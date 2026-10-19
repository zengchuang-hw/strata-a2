cmake_minimum_required(VERSION 3.20)
project(strata_a2 LANGUAGES CXX)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
execute_process(COMMAND "${Python3_EXECUTABLE}" -c
  "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_PREFIX}")
find_package(Torch REQUIRED)
execute_process(COMMAND "${Python3_EXECUTABLE}" -c
  "import pathlib,torch_npu; print(pathlib.Path(torch_npu.__file__).parent)"
  OUTPUT_VARIABLE TORCH_NPU_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
find_library(TORCH_NPU_LIBRARY torch_npu PATHS "${TORCH_NPU_ROOT}/lib" REQUIRED NO_DEFAULT_PATH)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib" REQUIRED NO_DEFAULT_PATH)
find_path(CANN_INCLUDE_DIR acl/acl_base_rt.h HINTS
  "$ENV{ASCEND_HOME_PATH}/include" "$ENV{ASCEND_TOOLKIT_HOME}/include"
  "/usr/local/Ascend/ascend-toolkit/latest/include" REQUIRED)
Python3_add_library(_strata_native MODULE src/runtime.cpp WITH_SOABI)
target_compile_features(_strata_native PRIVATE cxx_std_17)
target_include_directories(_strata_native PRIVATE "${CMAKE_SOURCE_DIR}/include"
  "${TORCH_NPU_ROOT}/include" "${CANN_INCLUDE_DIR}")
target_link_libraries(_strata_native PRIVATE "${TORCH_LIBRARIES}" "${TORCH_PYTHON_LIBRARY}" "${TORCH_NPU_LIBRARY}")
set_property(TARGET _strata_native PROPERTY BUILD_RPATH "${TORCH_NPU_ROOT}/lib;${TORCH_INSTALL_PREFIX}/lib")
separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(_strata_native PRIVATE ${TORCH_CXX_FLAGS_LIST} -O3 -Wall -Wextra)
add_executable(strata_run src/main.cpp)
set_property(TARGET strata_run PROPERTY OUTPUT_NAME "strata-run")
target_compile_definitions(strata_run PRIVATE
  STRATA_BOOTSTRAP_DIR="${CMAKE_SOURCE_DIR}/python/strata"
  STRATA_PYTHON_EXECUTABLE="${Python3_EXECUTABLE}"
  STRATA_NATIVE_BUILD="${CMAKE_CURRENT_BINARY_DIR}")
target_compile_features(strata_run PRIVATE cxx_std_17)
target_include_directories(strata_run PRIVATE ${TORCH_INCLUDE_DIRS} "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(strata_run PRIVATE Python3::Python ${CMAKE_DL_LIBS})
target_compile_options(strata_run PRIVATE ${TORCH_CXX_FLAGS_LIST} -O3 -Wall -Wextra)
