add_executable(tuning_advanced_end end.cpp)
target_link_libraries(tuning_advanced_end Kokkos::kokkos)
install(TARGETS tuning_advanced_end)
