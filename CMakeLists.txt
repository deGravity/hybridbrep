cmake_minimum_required(VERSION 3.14)
project(hybridbrep_cpp)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)

include(FetchContent)
FetchContent_Declare(
  brloader
  GIT_REPOSITORY   https://github.com/deGravity/breploader.git
  GIT_TAG   v0.2
)

FetchContent_MakeAvailable(brloader)

pybind11_add_module(hybridbrep_cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/hybridbrep.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/cpp/implicit_part.h
	${CMAKE_CURRENT_SOURCE_DIR}/cpp/implicit_part.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/hybrid_part.h
	${CMAKE_CURRENT_SOURCE_DIR}/cpp/hybrid_part.cpp
)

target_link_libraries(hybridbrep_cpp PUBLIC breploader Eigen3::Eigen)