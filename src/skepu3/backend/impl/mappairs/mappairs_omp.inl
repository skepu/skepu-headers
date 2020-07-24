/*! \file map_omp.inl
 *  \brief Contains the definitions of OpenMP specific member functions for the MapPairs skeleton.
 */

#ifdef SKEPU_OPENMP

#include <omp.h>

namespace skepu
{
	namespace backend
	{
		template<size_t Varity, size_t Harity, typename MapPairsFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... OI, size_t... VEI, size_t... HEI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapPairs<Varity, Harity, MapPairsFunc, CUDAKernel, CLKernel>
		::OMP(size_t Vsize, size_t Hsize, pack_indices<OI...>, pack_indices<VEI...>, pack_indices<HEI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			DEBUG_TEXT_LEVEL1("OpenMP MapPairs: hsize = " << Hsize << ", vsize = " << Vsize);
			
			// Sync with device data
			pack_expand((get<OI, CallArgs...>(args...).getParent().invalidateDeviceData(), 0)...);
			pack_expand((get<HEI, CallArgs...>(args...).getParent().updateHost(), 0)...);
			pack_expand((get<VEI, CallArgs...>(args...).getParent().updateHost(), 0)...);
			pack_expand((get<AI, CallArgs...>(args...).getParent().updateHost(hasReadAccess(MapPairsFunc::anyAccessMode[AI-Varity-Harity])), 0)...);
			pack_expand((get<AI, CallArgs...>(args...).getParent().invalidateDeviceData(hasWriteAccess(MapPairsFunc::anyAccessMode[AI-Varity-Harity])), 0)...);
			
#pragma omp parallel for schedule(runtime)
			for (size_t i = 0; i < Vsize; ++i)
			{
				for (size_t j = 0; j < Hsize; ++j)
				{
					Index2D index{ i, j };
					auto res = F::forward(MapPairsFunc::OMP, index,
						get<VEI, CallArgs...>(args...)(i)...,
						get<HEI, CallArgs...>(args...)(j)...,
						get<AI, CallArgs...>(args...).hostProxy(std::get<AI-Varity-Harity-outArity>(typename MapPairsFunc::ProxyTags{}), index)...,
						get<CI, CallArgs...>(args...)...);
					std::tie(get<OI, CallArgs...>(args...)(i, j)...) = res;
				}
			}
		}
	}
}

#endif // SKEPU_OPENMP
