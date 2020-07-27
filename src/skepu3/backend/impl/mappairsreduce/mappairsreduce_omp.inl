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
		template<size_t Varity, size_t Harity, typename MapPairsFunc, typename ReduceFunc, typename CUDAKernel, typename CUDAReduceKernel, typename CLKernel>
		template<size_t... OI, size_t... VEI, size_t... HEI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapPairsReduce<Varity, Harity, MapPairsFunc, ReduceFunc, CUDAKernel, CUDAReduceKernel, CLKernel>
		::OMP(size_t Vsize, size_t Hsize, pack_indices<OI...>, pack_indices<VEI...>, pack_indices<HEI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			DEBUG_TEXT_LEVEL1("OpenMP MapPairsReduce: hsize = " << Hsize << ", vsize = " << Vsize);
			
			// Sync with device data
			pack_expand((get<HEI, CallArgs...>(args...).getParent().updateHost(), 0)...);
			pack_expand((get<VEI, CallArgs...>(args...).getParent().updateHost(), 0)...);
			pack_expand((get<AI, CallArgs...>(args...).getParent().updateHost(hasReadAccess(MapPairsFunc::anyAccessMode[AI-Varity-Harity])), 0)...);
			pack_expand((get<AI, CallArgs...>(args...).getParent().invalidateDeviceData(hasWriteAccess(MapPairsFunc::anyAccessMode[AI-Varity-Harity])), 0)...);
			
			if (this->m_mode == ReduceMode::RowWise)
#pragma omp parallel for schedule(runtime)
				for (size_t i = 0; i < Vsize; ++i)
				{
					pack_expand((get<OI>(args...)(i) = get_or_return<OI>(this->m_start), 0)...);
					for (size_t j = 0; j < Hsize; ++j)
					{
						auto index = Index2D { i, j };
						auto temp = F::forward(MapPairsFunc::OMP, Index2D { i, j },
							get<VEI, CallArgs...>(args...)(i)...,
							get<HEI, CallArgs...>(args...)(j)...,
							get<AI, CallArgs...>(args...).hostProxy()...,
							get<CI, CallArgs...>(args...)...
						);
						pack_expand((get<OI>(args...)(i) = ReduceFunc::CPU(get<OI>(args...)(i), get_or_return<OI>(temp)), 0)...);
					}
				}
			else if (this->m_mode == ReduceMode::ColWise)
#pragma omp parallel for schedule(runtime)
				for (size_t j = 0; j < Hsize; ++j) // TODO: optimize?
				{
					pack_expand((get<OI>(args...)(j) = get_or_return<OI>(this->m_start), 0)...);
					for (size_t i = 0; i < Vsize; ++i)
					{
						auto index = Index2D { i, j };
						auto temp = F::forward(MapPairsFunc::OMP, Index2D { i, j },
							get<VEI, CallArgs...>(args...)(i)...,
							get<HEI, CallArgs...>(args...)(j)...,
							get<AI, CallArgs...>(args...).hostProxy()...,
							get<CI, CallArgs...>(args...)...
						);
						pack_expand((get<OI>(args...)(j) = ReduceFunc::CPU(get<OI>(args...)(j), get_or_return<OI>(temp)), 0)...);
					}
				}
		}
		
	} // namespace backend
} // namespace skepu

#endif
