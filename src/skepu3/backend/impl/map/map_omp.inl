/*! \file map_omp.inl
 *  \brief Contains the definitions of OpenMP specific member functions for the Map skeleton.
 */

#ifdef SKEPU_OPENMP

#include <omp.h>

namespace skepu
{
	namespace backend
	{
		template<size_t arity, typename MapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs> 
		void Map<arity, MapFunc, CUDAKernel, CLKernel>
		::OMP(size_t size, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			DEBUG_TEXT_LEVEL1("OpenMP Map: size = " << size);
			static constexpr auto proxy_tags = typename MapFunc::ProxyTags{};
			
			// Sync with device data
			pack_expand((get<EI, CallArgs...>(args...).getParent().updateHost(), 0)...);
			pack_expand((get<AI, CallArgs...>(args...).getParent().updateHost(hasReadAccess(MapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			pack_expand((get<AI, CallArgs...>(args...).getParent().invalidateDeviceData(hasWriteAccess(MapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			pack_expand((get<OI, CallArgs...>(args...).getParent().invalidateDeviceData(), 0)...);
			
#pragma omp parallel for schedule(runtime)
			for (size_t i = 0; i < size; ++i)
			{
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

#endif // SKEPU_OPENMP
