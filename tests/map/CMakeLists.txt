if(SKEPU_HEADERS_MPI)
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
	add_executable(starpu_mappairs starpu_mappairs.cpp)
	target_link_libraries(starpu_mappairs
		PRIVATE
			catch2_main SkePU::SkePU MPI::MPI_CXX PkgConfig::STARPU OpenMP::OpenMP_CXX)

	add_test(NAME starpu_mappairs_seq
		COMMAND starpu_mappairs)
	add_mpi_test(starpu_mappairs_par starpu_mappairs)

	add_executable(starpu_mappairsreduce starpu_mappairsreduce.cpp)
	target_link_libraries(starpu_mappairsreduce
		PRIVATE
			catch2_main SkePU::SkePU MPI::MPI_CXX PkgConfig::STARPU OpenMP::OpenMP_CXX)

	add_test(NAME starpu_mappairsreduce_seq
		COMMAND starpu_mappairsreduce)
	add_mpi_test(starpu_mappairsreduce_par starpu_mappairsreduce)
endif()
