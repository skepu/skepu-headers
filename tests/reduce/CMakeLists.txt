add_executable(starpu_reduce1d_elwise starpu_reduce1d_elwise.cpp)
target_link_libraries(starpu_reduce1d_elwise
	PRIVATE
		catch2_main SkePU::SkePU MPI::MPI_CXX PkgConfig::STARPU OpenMP::OpenMP_CXX)

add_executable(starpu_reduce1d_rowwise starpu_reduce1d_rowwise.cpp)
target_link_libraries(starpu_reduce1d_rowwise
	PRIVATE
		catch2_main SkePU::SkePU MPI::MPI_CXX PkgConfig::STARPU OpenMP::OpenMP_CXX)

add_executable(starpu_reduce1d_colwise starpu_reduce1d_colwise.cpp)
target_link_libraries(starpu_reduce1d_colwise
	PRIVATE
		catch2_main SkePU::SkePU MPI::MPI_CXX PkgConfig::STARPU OpenMP::OpenMP_CXX)

add_executable(starpu_reduce2d starpu_reduce2d.cpp)
target_link_libraries(starpu_reduce2d
	PRIVATE
		catch2_main SkePU::SkePU MPI::MPI_CXX PkgConfig::STARPU OpenMP::OpenMP_CXX)

add_test(NAME starpu_reduce1d_elwise_seq
	COMMAND starpu_reduce1d_elwise)
add_mpi_test(starpu_reduce1d_elwise_par starpu_reduce1d_elwise)

add_test(NAME starpu_reduce1d_rowwise_seq
	COMMAND starpu_reduce1d_rowwise)
add_mpi_test(starpu_reduce1d_rowwise_par starpu_reduce1d_rowwise)

add_test(NAME starpu_reduce1d_colwise_seq
	COMMAND starpu_reduce1d_colwise)
add_mpi_test(starpu_reduce1d_colwise_par starpu_reduce1d_colwise)

add_test(NAME starpu_reduce2d_seq
	COMMAND starpu_reduce2d)
add_mpi_test(starpu_reduce2d_par starpu_reduce2d)
