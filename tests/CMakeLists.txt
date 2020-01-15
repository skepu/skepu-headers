find_package(Catch2 REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(PkgConfig REQUIRED)

pkg_check_modules(STARPU  REQUIRED IMPORTED_TARGET
	starpu-1.3 starpumpi-1.3)

add_subdirectory(catch2_main)

# Use add_mpi_test(<name> <command>) to get a parallel mpi test added.
# Will make sure that mpirun is called on the executable and that STARPU_NCPU
# is set to one to reduce execution time. (One thread OpenMP execution within
# SkePU StarPU MPI backend.
macro(add_mpi_test name command)
	add_test(NAME ${name}
		COMMAND ${MPIEXEC_EXECUTABLE} ./${command})
	set_tests_properties(${name} PROPERTIES ENVIRONMENT
		STARPU_NCPU=1)
endmacro()

####################
# Add tests below. #
####################

# Containers
add_subdirectory(matrix)
add_subdirectory(vector)
add_subdirectory(tensor3)

# Skeletons
add_subdirectory(map)
add_subdirectory(reduce)
