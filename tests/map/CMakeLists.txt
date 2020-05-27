add_executable(starpu_map starpu_map.cpp)
target_link_libraries(starpu_map
	PRIVATE
		catch2_main SkePU::SkePU MPI::MPI_CXX PkgConfig::STARPU OpenMP::OpenMP_CXX)

add_test(NAME starpu_map_seq
	COMMAND starpu_map)
add_mpi_test(starpu_map_par starpu_map)

add_executable(starpu_mapreduce starpu_mapreduce.cpp)
target_link_libraries(starpu_mapreduce
	PRIVATE
		catch2_main SkePU::SkePU MPI::MPI_CXX PkgConfig::STARPU OpenMP::OpenMP_CXX)

add_test(NAME starpu_mapreduce_seq
	COMMAND starpu_mapreduce)
add_mpi_test(starpu_mapreduce_par starpu_mapreduce)
