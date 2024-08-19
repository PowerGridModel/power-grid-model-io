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

find_package(Eigen3 CONFIG REQUIRED)
find_package(Boost REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(msgpack-cxx REQUIRED)

add_subdirectory(power-grid-model)

message(STATUS "CMAKE_SOURCE_DIR is set to: ${CMAKE_SOURCE_DIR}")

add_executable(PgmVnfConverter src/main.cpp)

target_link_libraries(PgmVnfConverter PRIVATE power_grid_model)
