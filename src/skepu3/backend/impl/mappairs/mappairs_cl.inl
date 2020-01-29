/*! \file map_cl.inl
 *  \brief Contains the definitions of OpenCL specific member functions for the Map skeleton.
 */

#ifdef SKEPU_OPENCL

namespace skepu
{
	namespace backend
	{
		template<size_t Varity, size_t Harity, typename MapPairsFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... VEI, size_t... HEI, size_t... AI, size_t... CI, typename Iterator, typename... CallArgs> 
		void MapPairs<Varity, Harity, MapPairsFunc, CUDAKernel, CLKernel>
		::mapNumDevices_CL(size_t startIdx, size_t numDevices, size_t Vsize, size_t Hsize, pack_indices<VEI...>, pack_indices<HEI...>, pack_indices<AI...>, pack_indices<CI...>, Iterator res, CallArgs&&... args)
		{
			constexpr size_t numKernelArgs = numArgs + 3;
			const size_t numElemPerSlice = (Vsize * Hsize) / numDevices;
			const size_t rest = (Vsize * Hsize) % numDevices;
			
			std::vector<DeviceMemPointer_CL<T>*> outMemP(numDevices);
			
			for (size_t i = 0; i < numDevices; ++i)
			{
				Device_CL *device = this->m_environment->m_devices_CL[i];
				const size_t numElem = numElemPerSlice + ((i == numDevices-1) ? rest : 0);
				const size_t baseIndex = startIdx + i * numElemPerSlice;
				
				pack_expand((get<VEI, CallArgs...>(args...).getParent().updateDevice_CL(get<VEI, CallArgs...>(args...).getAddress() + baseIndex, numElem, device, false), 0)...);
				pack_expand((get<HEI, CallArgs...>(args...).getParent().updateDevice_CL(get<HEI, CallArgs...>(args...).getAddress() + baseIndex, numElem, device, false), 0)...);
				pack_expand((get<AI, CallArgs...>(args...).getParent().updateDevice_CL(get<AI, CallArgs...>(args...).getAddress(), get<AI, CallArgs...>(args...).size(), device, false), 0)...);
				outMemP[i] = res.getParent().updateDevice_CL(res.getAddress() + baseIndex, numElem, device, false);
			}
			
			DEBUG_TEXT_LEVEL1("OpenCL Map: numDevices " << numDevices);
			
			for (size_t i = 0; i < numDevices; ++i)
			{
				Device_CL *device = this->m_environment->m_devices_CL[i];
				const size_t numElem = numElemPerSlice + ((i == numDevices-1) ? rest : 0);
				const size_t numThreads = std::min(this->m_selected_spec->GPUThreads(), numElem);
				const size_t numBlocks = std::min(numElem / numThreads + (numElem % numThreads == 0 ? 0 : 1), this->m_selected_spec->GPUBlocks());
				const size_t baseIndex = startIdx + i * numElemPerSlice;
				
			//	DEBUG_TEXT_LEVEL1("OpenCL Map: device " << i << ", numElem = " << numElem << ", numBlocks = " << numBlocks << ", numThreads = " << numThreads);
				
				// Copies the elements to the device
				auto VelwiseMemP = std::make_tuple(get<VEI, CallArgs...>(args...).getParent().updateDevice_CL(get<VEI, CallArgs...>(args...).getAddress() + baseIndex, numElem, device, true)...);
				auto HelwiseMemP = std::make_tuple(get<HEI, CallArgs...>(args...).getParent().updateDevice_CL(get<HEI, CallArgs...>(args...).getAddress() + baseIndex, numElem, device, true)...);
				auto anyMemP = std::make_tuple(cl_helpers::randomAccessArg(get<AI, CallArgs...>(args...).getParent(), device, hasReadAccess(MapPairsFunc::anyAccessMode[AI-Varity-Harity]))...);
				
				CLKernel::map(
					i, numThreads, numBlocks * numThreads,
					std::get<VEI>(VelwiseMemP)...,
					std::get<HEI-Varity>(HelwiseMemP)...,
					std::get<AI-Varity-Harity>(anyMemP)...,
					get<CI, CallArgs...>(args...)...,
					outMemP[i],
					res.getParent().total_cols(),
					numElem,
					Vsize, Hsize,
					baseIndex
				);
				
				// Make sure the data is marked as changed by the device
				pack_expand((std::get<1>(std::get<AI-Varity-Harity>(anyMemP))->changeDeviceData(hasWriteAccess(MapPairsFunc::anyAccessMode[AI-Varity-Harity])), 0)...);
				outMemP[i]->changeDeviceData();
			}
		}
		
		
		template<size_t Varity, size_t Harity, typename MapPairsFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... VEI, size_t... HEI, size_t... AI, size_t... CI, typename Iterator, typename... CallArgs> 
		void MapPairs<Varity, Harity, MapPairsFunc, CUDAKernel, CLKernel>
		::CL(size_t startIdx, size_t Vsize, size_t Hsize, pack_indices<VEI...> vei, pack_indices<HEI...> hei, pack_indices<AI...> ai, pack_indices<CI...> ci, Iterator res, CallArgs&&... args)
		{
			DEBUG_TEXT_LEVEL1("OpenCL MapPairs: Vsize = " << Vsize << ", Hsize = " << Hsize << ", maxDevices = " << this->m_selected_spec->devices()
				<< ", maxBlocks = " << this->m_selected_spec->GPUBlocks() << ", maxThreads = " << this->m_selected_spec->GPUThreads());
			
			const size_t numDevices = std::min(this->m_selected_spec->devices(), this->m_environment->m_devices_CL.size());
			
			// Same method is invoked no matter how many GPUs we use
			this->mapNumDevices_CL(startIdx, numDevices, Vsize, Hsize, vei, hei, ai, ci, res, args...);
		}
		
	} // namespace backend
} // namespace skepu

#endif
