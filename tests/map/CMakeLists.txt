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

	# Apparently there is an issue where -Xcompiler is not added before propagated
	# cflags in cmake 3.10. Setting version 3.15 in the hope that it's fixed at
	# that point.
	if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.15)
		add_executable(starpu_map_cu starpu_map.cu)
		target_link_libraries(starpu_map_cu
			PRIVATE
				catch2_main
				SkePU::SkePU
				MPI::MPI_CXX
				PkgConfig::STARPU
				OpenMP::OpenMP_CXX)
		add_test(NAME starpu_map_seq_cu
			COMMAND starpu_map_cu)
		add_mpi_test(starpu_map_par_cu starpu_map_cu)
	endif()
endif()
