add_executable(instances_begin begin.cpp)
target_link_libraries(instances_begin Kokkos::kokkos)
install(TARGETS instances_begin)
