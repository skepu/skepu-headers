#pragma once

#include <vector>
// Mockup of the cluster interface, to allow the SkePU precompiler to
// compile the code.

namespace skepu2
{
	namespace cluster
	{
		static size_t
		mpi_rank() {
			return 0;
		};
		static size_t
		mpi_size() {
			return 1;
		};
		static size_t
		mpi_tag() {
			return 0;
		};
	}
}
