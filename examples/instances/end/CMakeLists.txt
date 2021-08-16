add_executable(instances_end end.cpp)
target_link_libraries(instances_end Kokkos::kokkos)
install(TARGETS instances_end)
