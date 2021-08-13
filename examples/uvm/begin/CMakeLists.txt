add_executable(uvm_begin begin.cpp)
target_link_libraries(uvm_begin Kokkos::kokkos)
install(TARGETS uvm_begin)
configure_file(uvm_caliper.in ${CMAKE_CURRENT_BINARY_DIR}/uvm_caliper)
install(
	FILES ${CMAKE_CURRENT_BINARY_DIR}/uvm_caliper 
	DESTINATION bin 
	PERMISSIONS 
	OWNER_EXECUTE OWNER_WRITE OWNER_READ 
	GROUP_EXECUTE GROUP_READ 
	WORLD_EXECUTE WORLD_READ)
