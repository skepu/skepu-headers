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
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename ...CallArgs>
		typename MapFunc::Ret MapReduce<arity, MapFunc, ReduceFunc, CUDAKernel, CUDAReduceKernel, CLKernel>
		::OMP(size_t size, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args)
		{
			// Sync with device data
			pack_expand((get<EI, CallArgs...>(args...).getParent().updateHost(), 0)...);
			pack_expand((get<AI, CallArgs...>(args...).getParent().updateHost(hasReadAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);
			pack_expand((get<AI, CallArgs...>(args...).getParent().invalidateDeviceData(hasWriteAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);
			
			std::vector<Ret> parsums(std::min<size_t>(size, omp_get_max_threads()));
			bool first = true;
			
			// Perform Map and partial Reduce with OpenMP
#pragma omp parallel for schedule(runtime) firstprivate(first)
			for (size_t i = 0; i < size; ++i)
			{
				size_t myid = omp_get_thread_num();
				auto index = (get<0, CallArgs...>(args...) + i).getIndex();
				Ret tempMap = F::forward(MapFunc::OMP,
					index,
					get<EI, CallArgs...>(args...)(i)...,
					get<AI, CallArgs...>(args...).hostProxy()...,
					get<CI, CallArgs...>(args...)...
				);
				if (first) 
				{
					parsums[myid] = tempMap;
					first = false;
				}
				else
					pack_expand((get_or_return<OI>(parsums[myid]) = ReduceFunc::OMP(get_or_return<OI>(parsums[myid]), get_or_return<OI>(tempMap)), 0)...);
			}
			
			// Final Reduce sequentially
			for (Ret &parsum : parsums)
				pack_expand((get_or_return<OI>(res) = ReduceFunc::OMP(get_or_return<OI>(res), get_or_return<OI>(parsum)), 0)...);
			
			return res;
		}
		
		
		
		template<size_t arity, typename MapFunc, typename ReduceFunc, typename CUDAKernel, typename CUDAReduceKernel, typename CLKernel>
		template<size_t... OI, size_t... AI, size_t... CI, typename ...CallArgs>
		typename MapFunc::Ret MapReduce<arity, MapFunc, ReduceFunc, CUDAKernel, CUDAReduceKernel, CLKernel>
		::OMP(size_t size, pack_indices<OI...>, pack_indices<>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args)
		{
			// Sync with device data
			pack_expand((get<AI, CallArgs...>(args...).getParent().updateHost(hasReadAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);
			pack_expand((get<AI, CallArgs...>(args...).getParent().invalidateDeviceData(hasWriteAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);
			
			std::vector<Ret> parsums(std::min<size_t>(size, omp_get_max_threads()));
			bool first = true;
			
			// Perform Map and partial Reduce with OpenMP
#pragma omp parallel for schedule(runtime) firstprivate(first)
			for (size_t i = 0; i < size; ++i)
			{
				size_t myid = omp_get_thread_num();
				auto index = make_index(defaultDim{}, i, this->default_size_j, this->default_size_k, this->default_size_l);
				Ret tempMap = F::forward(MapFunc::OMP,
					index,
					get<AI, CallArgs...>(args...).hostProxy()...,
					get<CI, CallArgs...>(args...)...
				);
				if (first) 
				{
					parsums[myid] = tempMap;
					first = false;
				}
				else
					pack_expand((get_or_return<OI>(parsums[myid]) = ReduceFunc::OMP(get_or_return<OI>(parsums[myid]), get_or_return<OI>(tempMap)), 0)...);
			}
			
			// Final Reduce sequentially
			for (Ret &parsum : parsums)
				pack_expand((get_or_return<OI>(res) = ReduceFunc::OMP(get_or_return<OI>(res), get_or_return<OI>(parsum)), 0)...);
			
			return res;
		}
		
	} // namespace backend
} // namespace skepu

#endif
