cmake_minimum_required(VERSION 3.16)
project(kokkos_tools_examples)

add_subdirectory(kokkos)
cmake_policy(SET CMP0077 NEW)
set(WITH_KOKKOS ON CACHE BOOL "")
set(BUILD_TESTING OFF)
if(Kokkos_ENABLE_CUDA)
	set(WITH_CUPTI ON CACHE BOOL "")
endif() # Kokkos_ENABLE_CUDA
add_subdirectory(caliper)

add_subdirectory(examples)
