add_executable(tuning_begin_advanced begin.cpp)
target_link_libraries(tuning_begin_advanced Kokkos::kokkos)
install(TARGETS tuning_begin_advanced)
