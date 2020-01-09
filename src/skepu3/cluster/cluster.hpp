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
				starpu_conf conf;
				int mpi_rank;
				int mpi_size;
				int mpi_tag {1};

				internal_state() {
					starpu_conf_init(&conf);
					conf.single_combined_worker = 1;
					if(conf.ncpus > 1)
						conf.sched_policy_name = "peager";
					// Not using starpu_mpi_init_conf because that makes
					// starpu_mpi_shutdown segfault.
					assert(!starpu_init(&conf));
					starpu_mpi_init(NULL, NULL, 1);

					mpi_rank = starpu_mpi_world_rank();
					mpi_size = starpu_mpi_world_size();
				};

				~internal_state() {
					starpu_mpi_wait_for_all(MPI_COMM_WORLD);
					starpu_mpi_shutdown();
					// StarPU shutdown is apparently not MPI safe when using performance
					// models.. However, starpu_mpi_shutdown should suffice.
					//starpu_shutdown();
				};
			};
		}
	}
}

namespace skepu {
	namespace cluster {
		namespace state {

			inline internal_state & s() {
				static internal_state g_state;
				return g_state;
			}
		}

		static size_t
		mpi_rank() {
			return state::s().mpi_rank;
		};

		static size_t
		mpi_size() {
			return state::s().mpi_size;
		};

		// Return a new mpi_tag for use with a handle, as a unique
		// identifier for use in registering data handles. This function
		// *must* be called coherently across all ranks.
		static size_t
		mpi_tag() {
			return state::s().mpi_tag++;
		};

		static auto
		starpu_ncpus()
		-> int
		{
			return state::s().conf.ncpus;
		}

		static auto
		barrier()
		-> void
		{
			starpu_mpi_wait_for_all(MPI_COMM_WORLD);
		}
	}

	inline void wait_for_all_tasks() {
		if(starpu_task_nsubmitted()) {
			starpu_task_wait_for_all();
		}
	}
}

#endif /* CLUSTER_HPP */
