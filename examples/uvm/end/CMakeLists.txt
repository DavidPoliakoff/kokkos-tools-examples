add_executable(uvm_end end.cpp)
target_link_libraries(uvm_end Kokkos::kokkos)
install(TARGETS uvm_end)
