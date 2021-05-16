/*! \file map_cpu.inl
 *  \brief Contains the definitions of CPU specific member functions for the Map skeleton.
 */

namespace skepu
{
	namespace backend
	{
		template<size_t arity, typename MapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs, REQUIRES_B(!MapFunc::usesPRNG && true && sizeof...(CallArgs) > 0)> 
		void Map<arity, MapFunc, CUDAKernel, CLKernel> 
		::CPU(size_t size, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			DEBUG_TEXT_LEVEL1("CPU Map: size = " << size);
			
			// Sync with device data
			pack_expand((get<EI>(args...).getParent().updateHost(), 0)...);
			pack_expand((get<AI>(args...).getParent().updateHost(hasReadAccess(MapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			pack_expand((get<AI>(args...).getParent().invalidateDeviceData(hasWriteAccess(MapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			pack_expand((get<OI>(args...).getParent().invalidateDeviceData(), 0)...);
			
			for (size_t i = 0; i < size; ++i)
			{
				auto index = (std::get<0>(std::make_tuple(get<OI>(args...).begin()...)) + i).getIndex();
				auto res = F::forward(MapFunc::CPU, index,
					get<EI>(args...)(i)..., 
					get<AI>(args...).hostProxy(std::get<AI-arity-outArity>(typename MapFunc::ProxyTags{}), index)...,
					get<CI>(args...)...
				);
				SKEPU_VARIADIC_RETURN(get<OI>(args...)(i)..., res);
			}
		}
		
		
		
		// PRNG
		
		template<size_t arity, typename MapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs, REQUIRES_B(!!MapFunc::usesPRNG && sizeof...(CallArgs) > 0)> 
		void Map<arity, MapFunc, CUDAKernel, CLKernel> 
		::CPU(size_t size, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			DEBUG_TEXT_LEVEL1("CPU Map: size = " << size);
			
			// Sync with device data
			pack_expand((get<EI>(args...).getParent().updateHost(), 0)...);
			pack_expand((get<AI>(args...).getParent().updateHost(hasReadAccess(MapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			pack_expand((get<AI>(args...).getParent().invalidateDeviceData(hasWriteAccess(MapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			pack_expand((get<OI>(args...).getParent().invalidateDeviceData(), 0)...);
			
			if (this->m_prng == nullptr)
				SKEPU_ERROR("No random stream set in skeleton instance");
			
			auto random = this->m_prng->template asRandom<MapFunc::randomCount>(size);
			
			for (size_t i = 0; i < size; ++i)
			{
				auto index = (std::get<0>(std::make_tuple(get<OI>(args...).begin()...)) + i).getIndex();
				auto res = F::forward(MapFunc::CPU, index, random,
					get<EI>(args...)(i)..., 
					get<AI>(args...).hostProxy(std::get<AI-arity-outArity>(typename MapFunc::ProxyTags{}), index)...,
					get<CI>(args...)...
				);
				SKEPU_VARIADIC_RETURN(get<OI>(args...)(i)..., res);
			}
		}
	}
}

