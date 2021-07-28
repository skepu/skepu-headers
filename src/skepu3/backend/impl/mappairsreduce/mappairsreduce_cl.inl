/*! \file mapreduce_cl.inl
 *  \brief Contains the definitions of OpenCL specific member functions for the MapReduce skeleton.
 */

#ifdef SKEPU_OPENCL

namespace skepu
{
	namespace backend
	{
		size_t closestPow2(size_t x)
		{
			size_t guess = 1;
			while (guess < x) { guess *= 2; }
			return guess;
		}
		
		
		template<size_t Varity, size_t Harity, typename MapPairsFunc, typename ReduceFunc, typename CUDAKernel, typename CUDAReduceKernel, typename CLKernel>
		template<size_t... OI, size_t... VEI, size_t... HEI, size_t... AI, size_t... CI, typename... CallArgs> 
		void MapPairsReduce<Varity, Harity, MapPairsFunc, ReduceFunc, CUDAKernel, CUDAReduceKernel, CLKernel>
		::mapPairsReduceNumDevices_CL(size_t numDevices, size_t startIdx, size_t Vsize, size_t Hsize, pack_indices<OI...>, pack_indices<VEI...>, pack_indices<HEI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			if (this->m_mode == ReduceMode::ColWise)
				std::swap(Vsize, Hsize);
			
			const size_t numRowsPerSlice = Vsize / numDevices;
			const size_t rest = Vsize % numDevices;
			
			for (size_t i = 0; i < numDevices; ++i)
			{
				Device_CL *device = this->m_environment->m_devices_CL[i];
				const size_t baseIndex = startIdx + i * numRowsPerSlice * Hsize;
				
				pack_expand((get<OI>(std::forward<CallArgs>(args)...).getParent().updateDevice_CL(get<OI>(std::forward<CallArgs>(args)...).getAddress() + baseIndex, Vsize, device, false), 0)...);
				pack_expand((get<VEI>(std::forward<CallArgs>(args)...).getParent().updateDevice_CL(get<VEI>(std::forward<CallArgs>(args)...).getAddress() + baseIndex, Vsize, device, false), 0)...);
				pack_expand((get<HEI>(std::forward<CallArgs>(args)...).getParent().updateDevice_CL(get<HEI>(std::forward<CallArgs>(args)...).getAddress() + baseIndex, Hsize, device, false), 0)...);
				pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().updateDevice_CL(get<AI>(std::forward<CallArgs>(args)...).getAddress(), get<AI>(std::forward<CallArgs>(args)...).size(), device, false), 0)...);
			}
			
			for (size_t i = 0; i < numDevices; ++i)
			{
				Device_CL *device = this->m_environment->m_devices_CL[i];
			//	const size_t numElem = (numRowsPerSlice + ((i == numDevices-1) ? rest : 0)) * Hsize;
				const size_t numThreads = std::min(this->m_selected_spec->GPUThreads(), closestPow2(Hsize));
				const size_t numBlocks = std::min(Vsize, this->m_selected_spec->GPUBlocks());
				const size_t baseIndex = startIdx + i * numRowsPerSlice * Hsize;
				
				DEBUG_TEXT_LEVEL1("OpenCL MapPairsReduce: device " << i << ", numElem = " << 0 << ", numBlocks = " << numBlocks << ", numThreads = " << numThreads);
				
				// Copies the elements to the device
				auto outMemP     = std::make_tuple(get<OI>(std::forward<CallArgs>(args)...).getParent().updateDevice_CL(get<OI>(std::forward<CallArgs>(args)...).getAddress() + baseIndex, Vsize, device, false)...);
				auto VelwiseMemP = std::make_tuple(get<VEI>(std::forward<CallArgs>(args)...).getParent().updateDevice_CL(get<VEI>(std::forward<CallArgs>(args)...).getAddress() + baseIndex, Vsize, device, true)...);
				auto HelwiseMemP = std::make_tuple(get<HEI>(std::forward<CallArgs>(args)...).getParent().updateDevice_CL(get<HEI>(std::forward<CallArgs>(args)...).getAddress() + baseIndex, Hsize, device, true)...);
				auto anyMemP = std::make_tuple(cl_helpers::randomAccessArg(get<AI>(std::forward<CallArgs>(args)...).getParent(), device, hasReadAccess(MapPairsFunc::anyAccessMode[AI-Varity-Harity]))...);
			
				size_t threads = std::min<size_t>(1, numBlocks * numThreads); // FIX
				auto random = this->template prepareRandom_CL<MapPairsFunc::randomCount>(1, threads);
				auto randomMemP = random.updateDevice_CL(random.getAddress(), threads, device, true);
				
				CLKernel::mapPairsReduce(
					i, numThreads, numBlocks * numThreads,
					std::get<OI>(outMemP)...,
					randomMemP,
					std::get<VEI-outArity>(VelwiseMemP)...,
					std::get<HEI-Varity-outArity>(HelwiseMemP)...,
					std::get<AI-Varity-Harity-outArity>(anyMemP)...,
					get<CI>(std::forward<CallArgs>(args)...)...,
					Vsize * Hsize,
					Vsize, Hsize,
					baseIndex,
					(this->m_mode == ReduceMode::ColWise) ? 1 : 0,
					sizeof(Ret) * numThreads
				);
				
				// Make sure the data is marked as changed by the device
				pack_expand((std::get<1>(std::get<AI-Varity-Harity-outArity>(anyMemP))->changeDeviceData(hasWriteAccess(MapPairsFunc::anyAccessMode[AI-Varity-Harity-outArity])), 0)...);
				pack_expand((std::get<OI>(outMemP)->changeDeviceData(), 0)...);
			}
		}
		
		
		template<size_t Varity, size_t Harity, typename MapPairsFunc, typename ReduceFunc, typename CUDAKernel, typename CUDAReduceKernel, typename CLKernel>
		template<size_t... OI, size_t... VEI, size_t... HEI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapPairsReduce<Varity, Harity, MapPairsFunc, ReduceFunc, CUDAKernel, CUDAReduceKernel, CLKernel>
		::CL(size_t startIdx, size_t Vsize, size_t Hsize, pack_indices<OI...> oi, pack_indices<VEI...> vei, pack_indices<HEI...> hei, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args)
		{
			DEBUG_TEXT_LEVEL1("OpenCL MapPairsReduce: Vsize = " << Vsize << ", Hsize = " << Hsize << ", maxDevices = " << this->m_selected_spec->devices()
				<< ", maxBlocks = " << this->m_selected_spec->GPUBlocks() << ", maxThreads = " << this->m_selected_spec->GPUThreads());
			
			const size_t numDevices = std::min(this->m_selected_spec->devices(), this->m_environment->m_devices_CL.size());
			
			// Same method is invoked no matter how many GPUs we use
			this->mapPairsReduceNumDevices_CL(numDevices, startIdx, Vsize, Hsize, oi, vei, hei, ai, ci, std::forward<CallArgs>(args)...);
		}
		
	} // end namespace backend
} // end namespace skepu

#endif
