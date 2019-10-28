add_executable(vector vector.cpp)
target_link_libraries(vector PRIVATE skepu-headers catch2_main)

add_test(NAME vector_test
	COMMAND vector)
