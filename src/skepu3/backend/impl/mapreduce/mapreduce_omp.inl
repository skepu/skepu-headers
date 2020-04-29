/*! \file mapreduce_omp.inl
*  \brief Contains the definitions of OpenMP specific member functions for the MapReduce skeleton.
*/

#ifdef SKEPU_OPENMP

#include <omp.h>
#include <iostream>
#include <vector>

namespace skepu
{
	namespace backend
	{
		template<size_t arity, typename MapFunc, typename ReduceFunc, typename CUDAKernel, typename CUDAReduceKernel, typename CLKernel>
		template<size_t... EI, size_t... AI, size_t... CI, typename ...CallArgs>
		typename ReduceFunc::Ret MapReduce<arity, MapFunc, ReduceFunc, CUDAKernel, CUDAReduceKernel, CLKernel>
		::OMP(size_t size, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args)
		{
			// Sync with device data
			pack_expand((get<EI, CallArgs...>(args...).getParent().updateHost(), 0)...);
			pack_expand((get<AI, CallArgs...>(args...).getParent().updateHost(hasReadAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);
			pack_expand((get<AI, CallArgs...>(args...).getParent().invalidateDeviceData(hasWriteAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);
			
			std::vector<Ret> parsums(this->m_selected_spec->CPUThreads(), this->m_start);
			
			// Perform Map and partial Reduce with OpenMP
#pragma omp parallel for schedule(runtime)
			for (size_t i = 0; i < size; ++i)
			{
				size_t myid = omp_get_thread_num();
				auto index = (get<0, CallArgs...>(args...) + i).getIndex();
				Temp tempMap = F::forward(MapFunc::OMP, index, get<EI, CallArgs...>(args...)(i)..., get<AI, CallArgs...>(args...).hostProxy()..., get<CI, CallArgs...>(args...)...);
				parsums[myid] = ReduceFunc::OMP(parsums[myid], tempMap);
			}
			
			// Final Reduce sequentially
			for (Ret &parsum : parsums)
				res = ReduceFunc::OMP(res, parsum);
			
			return res;
		}
		
		
		
		template<size_t arity, typename MapFunc, typename ReduceFunc, typename CUDAKernel, typename CUDAReduceKernel, typename CLKernel>
		template<size_t... AI, size_t... CI, typename ...CallArgs>
		typename ReduceFunc::Ret MapReduce<arity, MapFunc, ReduceFunc, CUDAKernel, CUDAReduceKernel, CLKernel>
		::OMP(size_t size, pack_indices<>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args)
		{
			// Sync with device data
			pack_expand((get<AI, CallArgs...>(args...).getParent().updateHost(hasReadAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);
			pack_expand((get<AI, CallArgs...>(args...).getParent().invalidateDeviceData(hasWriteAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);
			
			std::vector<Ret> parsums(this->m_selected_spec->CPUThreads(), this->m_start);
			
			// Perform Map and partial Reduce with OpenMP
#pragma omp parallel for schedule(runtime)
			for (size_t i = 0; i < size; ++i)
			{
				size_t myid = omp_get_thread_num();
				auto index = make_index(defaultDim{}, i, this->default_size_j, this->default_size_k, this->default_size_l);
				Temp tempMap = F::forward(MapFunc::OMP, index, get<AI, CallArgs...>(args...).hostProxy()..., get<CI, CallArgs...>(args...)...);
				parsums[myid] = ReduceFunc::OMP(parsums[myid], tempMap);
			}
			
			// Final Reduce sequentially
			for (Ret &parsum : parsums)
				res = ReduceFunc::OMP(res, parsum);
			
			return res;
		}
		
	} // namespace backend
} // namespace skepu

#endif
