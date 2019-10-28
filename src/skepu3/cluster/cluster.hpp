#ifndef CLUSTER_HPP
#define CLUSTER_HPP

#include <stdio.h>
#include <starpu.h>
#include <iostream>
#include <starpu_mpi.h>
#include <mpi.h>

namespace skepu {
	namespace cluster {
		namespace state {
			struct internal_state {
				int mpi_rank;
				int mpi_size;
				int mpi_tag {1};
				int mpi_provided_thread_support {};
				internal_state() {
					assert(!starpu_init(NULL));
					starpu_mpi_init(NULL, NULL, 1);

					starpu_mpi_comm_rank(MPI_COMM_WORLD, &mpi_rank);
					starpu_mpi_comm_size(MPI_COMM_WORLD, &mpi_size);
				};

				~internal_state() {
					MPI_Barrier(MPI_COMM_WORLD);
					starpu_mpi_shutdown();
					starpu_shutdown();
				};
			};
		}
	}
}

namespace skepu {
	namespace cluster {
		namespace state {

			inline internal_state * s() {
				static internal_state g_state;
				return &g_state;
			}
		}

		static size_t
		mpi_rank() {
			return state::s()->mpi_rank;
		};

		static size_t
		mpi_size() {
			return state::s()->mpi_size;
		};

		// Return a new mpi_tag for use with a handle, as a unique
		// identifier for use in registering data handles. This function
		// *must* be called coherently across all ranks.
		static size_t
		mpi_tag() {
			return state::s()->mpi_tag++;
		};
	}

	inline void wait_for_all_tasks() {
		if(starpu_task_nsubmitted()) {
			starpu_task_wait_for_all();
		}
	}
}

#endif /* CLUSTER_HPP */
