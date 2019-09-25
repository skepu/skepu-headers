#pragma once

// Deprecated operators are still accessible in this version 
#define SKEPU_ENABLE_DEPRECATED_OPERATOR

#if defined(SKEPU_OPENMP) && (defined(SKEPU_CUDA) || defined(SKEPU_OPENCL))
# define SKEPU_HYBRID

# if defined(SKEPU_CUDA) && !(defined(SKEPU_OPENCL) && defined(SKEPU_HYBRID_FORCE_OPENCL))
#  define SKEPU_HYBRID_USE_CUDA
# endif
#endif

/* TODO: Get rid of this hack if possible. */
#ifdef SKEPU_MERCURIUM_CUDA
#define __CUDACC__
#include <cuda.h>
#include <device_functions.h>
#undef __CUDACC__
#endif

#ifdef SKEPU_MPI_STARPU

#include "skepu2/cluster/cluster.hpp"
#include "skepu2/cluster/starpu_matrix_container.hpp"
#include "skepu2/cluster/task_creator.hpp"
#include "skepu2/cluster/matrix.hpp"
#include "skepu2/cluster/vec.hpp"
#include "skepu2/cluster/vector.hpp"
#include "skepu2/cluster/matrix_iterator.hpp"
#include "skepu2/cluster/vector_iterator.hpp"
#include "skepu2/cluster/access_mode.hpp"
#include "skepu2/cluster/map.hpp"
#include "skepu2/cluster/reduce1d.hpp"
#include "skepu2/backend/debug.h"

#else

#include "skepu2/cluster/cluster_fake.hpp"

#include "skepu2/impl/backend.hpp"
#include "skepu2/backend/helper_methods.h"
#include "skepu2/impl/common.hpp"
#include "skepu2/impl/timer.hpp"
#include "skepu2/backend/tuner.h"
#include "skepu2/backend/hybrid_tuner.h"

#ifndef SKEPU_PRECOMPILED

#include "skepu2/map.hpp"
#include "skepu2/reduce.hpp"
#include "skepu2/scan.hpp"
#include "skepu2/mapoverlap.hpp"
#include "skepu2/mapreduce.hpp"
#include "skepu2/mappairs.hpp"
#include "skepu2/call.hpp"

#else

#include "skepu2/backend/skeleton_base.h"
#include "skepu2/backend/map.h"
#include "skepu2/backend/reduce.h"
#include "skepu2/backend/mapreduce.h"
#include "skepu2/backend/scan.h"
#include "skepu2/backend/mapoverlap.h"
#include "skepu2/backend/mappairs.h"
#include "skepu2/backend/call.h"


#endif // SKEPU_PRECOMPILED
#endif // SKEPU_MPI_STARPU
