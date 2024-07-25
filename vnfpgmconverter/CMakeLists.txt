cmake_minimum_required(VERSION 3.23)

project(vnf_pgm_converter VERSION 0.1)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find the Power Grid Model C API library
# find_library(POWER_GRID_MODEL_LIB power_grid_model_c)
# set(POWER_GRID_MODEL_LIB "path")

set(ADDITIONAL_LIBRARY_PATH "path")

set(ADDITIONAL_INCLUDE_PATH "path")
set(SUB_INCLUDE_PATH "path")

message(STATUS "Additional Library Path: ${ADDITIONAL_LIBRARY_PATH}")
message(STATUS "Additional Include Path: ${ADDITIONAL_INCLUDE_PATH}")
message(STATUS "Sub Include Path: ${SUB_INCLUDE_PATH}")

find_library(POWER_GRID_MODEL_LIB NAMES power_grid_model_c PATHS ${ADDITIONAL_LIBRARY_PATH})

if (NOT POWER_GRID_MODEL_LIB)
    message(FATAL_ERROR "PowerGridModel library not found. Ensure it is in the specified path: ${ADDITIONAL_LIBRARY_PATH}")
else()
    message(STATUS "PowerGridModel library found: ${POWER_GRID_MODEL_LIB}")
endif()

include_directories(${CMAKE_SOURCE_DIR}/include ${ADDITIONAL_INCLUDE_PATH} ${SUB_INCLUDE_PATH})

message(STATUS "Source Directory Include Path: ${CMAKE_SOURCE_DIR}/include")

add_executable(PgmVnfConverter src/vnf_pgm_converter.cpp src/c_converter_api.cpp)

target_link_libraries(PgmVnfConverter ${POWER_GRID_MODEL_LIB})
