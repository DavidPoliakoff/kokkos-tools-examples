add_executable(tuning_begin begin.cpp)
target_link_libraries(tuning_begin Kokkos::kokkos)
install(TARGETS tuning_begin)
