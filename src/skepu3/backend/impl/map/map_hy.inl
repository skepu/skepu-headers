/*! \file map_hy.inl
 *  \brief Contains the definitions of Hybrid execution specific member functions for the Map skeleton.
 * 
 * The data is divided between the CPU and the GPU(s) as follows:
 * 
 *  i =  0           ...                 size
 *       ####CPU#####   ########GPU########
 * 
 * The GPU part might in turn split its part of the work between multiple GPUs.
 */

#ifdef SKEPU_HYBRID

#include <omp.h>
#include <iostream>
#include <functional>

namespace skepu
{
	namespace backend
	{
	
		template<size_t arity, typename MapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs> 
		void Map<arity, MapFunc, CUDAKernel, CLKernel>
		::Hybrid(size_t size, pack_indices<OI...> oi, pack_indices<EI...> ei, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args)
		{
			static constexpr auto proxy_tags = typename MapFunc::ProxyTags{};

			const float cpuPartitionSize = this->m_selected_spec->CPUPartitionRatio();
			const size_t cpuSize = cpuPartitionSize*size;
			const size_t gpuSize = size-cpuSize;
			size_t nthr = this->m_selected_spec->CPUThreads();

			DEBUG_TEXT_LEVEL1("Hybrid Map: size = " << size << " CPU partition: " << (100.0f*cpuPartitionSize) << "%");
			
			// If one partition is considered too small, fall back to GPU-only or CPU-only
			if(gpuSize < 32) { // Not smaller than a warp (=32 threads)
				DEBUG_TEXT_LEVEL1("Hybrid Map: Too small GPU size, fall back to CPU-only.");
				this->OMP(size, oi, ei, ai, ci, args...);
				return;
			}
			else if(cpuSize < nthr) {
				DEBUG_TEXT_LEVEL1("Hybrid Map: Too small CPU size, fall back to GPU-only.");
#ifdef SKEPU_HYBRID_USE_CUDA
				this->CUDA(0, size, oi, ei, ai, ci, args...);
#else
				this->CL(0, size, oi, ei, ai, ci, args...);
#endif
				return;
			}
			
			// Sync with device data
			pack_expand((get<EI, CallArgs...>(args...).getParent().updateHost(), 0)...);
			pack_expand((get<AI, CallArgs...>(args...).getParent().updateHost(hasReadAccess(MapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			pack_expand((get<AI, CallArgs...>(args...).getParent().invalidateDeviceData(hasWriteAccess(MapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			
			if(nthr < 2)
				nthr = 2; // Make sure to always have at least one CPU and one GPU thread
				
			omp_set_num_threads(nthr);
			const size_t numCPUThreads = nthr - 1; // One thread is used for GPU
			
#pragma omp parallel
			{
				size_t myId = omp_get_thread_num();
					
				if(myId == 0) {
					// Let first thread take care of GPU
#ifdef SKEPU_HYBRID_USE_CUDA
					this->CUDA(cpuSize, gpuSize, oi, ei, ai, ci, args...);
#else
					this->CL(cpuSize, gpuSize, oi, ei, ai, ci, args...);
#endif
				}
				else {
					myId--; // Reindex CPU threads 0...numCPUThreads

					const size_t blockSize = cpuSize/numCPUThreads;
					size_t workSize = blockSize;
					if(myId == (numCPUThreads-1) ) // Last CPU thread gets the rest
						workSize += cpuSize%numCPUThreads;
					
					const size_t first = blockSize*myId;
					const size_t last = blockSize*myId + workSize;
					
					for (size_t i = first; i < last; ++i)
					{
					//	res(i) = F::forward(MapFunc::OMP, (res + i).getIndex(), get<EI, CallArgs...>(args...)(i)..., get<AI, CallArgs...>(args...).hostProxy()..., get<CI, CallArgs...>(args...)...);
						auto index = (std::get<0>(std::make_tuple(get<OI, CallArgs...>(args...).begin()...)) + i).getIndex();
						auto res = F::forward(MapFunc::OMP, index,
							get<EI, CallArgs...>(args...)(i)..., 
							get<AI, CallArgs...>(args...).hostProxy(std::get<AI-arity-outArity>(proxy_tags), index)...,
							get<CI, CallArgs...>(args...)...
						);
						std::tie(get<OI, CallArgs...>(args...)(i)...) = res;
					}
				}
			}
		}
		
	} // namespace backend
} // namespace skepu

#endif
