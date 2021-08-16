add_executable(tuning_advanced_begin begin.cpp)
target_link_libraries(tuning_advanced_begin Kokkos::kokkos)
install(TARGETS tuning_advanced_begin)
