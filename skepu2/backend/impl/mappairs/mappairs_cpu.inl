/*! \file map_cpu.inl
 *  \brief Contains the definitions of CPU specific member functions for the MapPairs skeleton.
 */

namespace skepu2
{
	namespace backend
	{
		template<size_t Varity, size_t Harity, typename MapPairsFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... VEI, size_t... HEI, size_t... AI, size_t... CI, typename Iterator, typename... CallArgs> 
		void MapPairs<Varity, Harity, MapPairsFunc, CUDAKernel, CLKernel> 
		::CPU(size_t Hsize, size_t Vsize, pack_indices<VEI...>, pack_indices<HEI...>, pack_indices<AI...>, pack_indices<CI...>, Iterator res, CallArgs&&... args)
		{
			DEBUG_TEXT_LEVEL1("CPU MapPairs: size = " << size);
			
			// Sync with device data
			pack_expand((get<HEI, CallArgs...>(args...).getParent().updateHost(), 0)...);
			pack_expand((get<VEI, CallArgs...>(args...).getParent().updateHost(), 0)...);
			pack_expand((get<AI, CallArgs...>(args...).getParent().updateHost(hasReadAccess(MapPairsFunc::anyAccessMode[AI-Varity-Harity])), 0)...);
			pack_expand((get<AI, CallArgs...>(args...).getParent().invalidateDeviceData(hasWriteAccess(MapPairsFunc::anyAccessMode[AI-Varity-Harity])), 0)...);
			res.getParent().invalidateDeviceData();
			
			for (size_t i = 0; i < Vsize; ++i)
			{
				for (size_t j = 0; j < Hsize; ++j)
				{
					auto index = Index2D { i, j };
					res(i, j) = F::forward(MapPairsFunc::CPU, index, get<VEI, CallArgs...>(args...)(i)..., get<HEI, CallArgs...>(args...)(i)..., get<AI, CallArgs...>(args...).hostProxy()..., get<CI, CallArgs...>(args...)...);
				}
			}
		}
	}
}

