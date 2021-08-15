add_executable(tuning_end_advanced end.cpp)
target_link_libraries(tuning_end_advanced Kokkos::kokkos)
install(TARGETS tuning_end_advanced)
