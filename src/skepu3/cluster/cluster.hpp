#ifndef CLUSTER_HPP
#define CLUSTER_HPP

#include <starpu.h>
#include <starpu_mpi.h>

namespace skepu {
	namespace cluster {
		namespace state {
			struct internal_state {
				starpu_conf conf;
				int mpi_rank;
				int mpi_size;
				int mpi_tag {1};

				// Figure out howto enable all cores when not using STARPU_NCPU
				internal_state() {
					starpu_conf_init(&conf);
					conf.sched_policy_name = "peager";
					if(conf.ncpus > 1)
						conf.reserve_ncpus = 1;
					// Not using starpu_mpi_init_conf because that makes
					// starpu_mpi_shutdown segfault.
					// TODO:As the performance models in the skeletons are not static in
					// any way. Change that and it shall work fine.
					auto err = starpu_init(&conf);
					assert(!err);
					starpu_mpi_init(NULL, NULL, 1);

					mpi_rank = starpu_mpi_world_rank();
					mpi_size = starpu_mpi_world_size();
				};

				~internal_state() {
					starpu_mpi_shutdown();
					// TODO: See comment in constructor...
					//starpu_mpi_shutdown();
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
