/*! \file mappairsreduce.h
 *  \brief Contains a class declaration for the MapPairsReduce skeleton.
 */

#ifndef MAPPAIRSREDUCE_H
#define MAPPAIRSREDUCE_H

namespace skepu
{
	namespace backend
	{
		/*!
		*  \ingroup skeletons
		*/
		/*!
		*  \class MapPairsReduce
		*
		*  \brief A class representing the MapPairsReduce skeleton.
		*/
		template<size_t Varity, size_t Harity, typename MapPairsFunc, typename ReduceFunc, typename CUDAKernel, typename CUDAReduceKernel, typename CLKernel>
		class MapPairsReduce : public SkeletonBase
		{
		public:
			MapPairsReduce(CUDAKernel mappairsreduce, CUDAReduceKernel reduce)
			: m_cuda_kernel(mappairsreduce), m_cuda_reduce_kernel(reduce)
			{
#ifdef SKEPU_OPENCL
				CLKernel::initialize();
#endif
			}
			
			static constexpr auto skeletonType = SkeletonType::MapPairsReduce;
			using ResultArg = std::tuple<>;
			using ElwiseArgs = typename MapPairsFunc::ElwiseArgs;
			using ContainerArgs = typename MapPairsFunc::ContainerArgs;
			using UniformArgs = typename MapPairsFunc::UniformArgs;
			static constexpr bool prefers_matrix = MapPairsFunc::prefersMatrix;
			
		private:
			CUDAKernel m_cuda_kernel;
			CUDAReduceKernel m_cuda_reduce_kernel;
			size_t default_size_i;
			size_t default_size_j;
			
			static constexpr size_t numArgs = MapPairsFunc::totalArity - (MapPairsFunc::indexed ? 1 : 0);
			static constexpr size_t anyArity = std::tuple_size<typename MapPairsFunc::ContainerArgs>::value;
			
			static constexpr typename make_pack_indices<Varity, 0>::type Velwise_indices{};
			static constexpr typename make_pack_indices<Harity + Varity, Varity>::type Helwise_indices{};
			static constexpr typename make_pack_indices<Varity + Harity + anyArity, Varity + Harity>::type any_indices{};
			static constexpr typename make_pack_indices<numArgs, Varity + Harity + anyArity>::type const_indices{};
			
			using defaultDim = typename std::conditional<MapPairsFunc::indexed, index_dimension<typename MapPairsFunc::IndexType>, std::integral_constant<int, 1>>::type;
			using First = typename parameter_type<MapPairsFunc::indexed ? 1 : 0, decltype(&MapPairsFunc::CPU)>::type;
			
			using F = ConditionalIndexForwarder<MapPairsFunc::indexed, decltype(&MapPairsFunc::CPU)>;
			using Temp = typename MapPairsFunc::Ret;
			using Ret = typename ReduceFunc::Ret;
			
			Ret m_start{};
			ReduceMode m_mode = ReduceMode::RowWise;
			
#pragma mark - Backend agnostic
			
		public:
			void setStartValue(Ret val)
			{
				this->m_start = val;
			}
			
			void setDefaultSize(size_t i, size_t j = 0)
			{
				this->default_size_i = i;
				this->default_size_j = j;
			}
			
			void setReduceMode(ReduceMode mode)
			{
				this->m_mode = mode;
			}
			
			template<typename... Args>
			void tune(Args&&... args)
			{
				tuner::tune(*this, std::forward<Args>(args)...);
			}
			
			template<typename... CallArgs>
			Vector<Ret> &operator()(Vector<Ret> &res, CallArgs&&... args)
			{
				backendDispatch(Velwise_indices, Helwise_indices, any_indices, const_indices, res.size(), res, std::forward<CallArgs>(args)...);
				return res;
			}
			
			template<typename Iterator, typename... CallArgs, REQUIRES(is_skepu_iterator<Iterator, First>::value)>
			Iterator operator()(Iterator res, Iterator res_end, CallArgs&&... args)
			{
				backendDispatch(Velwise_indices, Helwise_indices, any_indices, const_indices, res_end - res, res, std::forward<CallArgs>(args)...);
				return res;
			}
			
		private:
			template<typename Result, size_t... VEI, size_t... HEI, size_t... AI, size_t... CI, typename... CallArgs>
			void backendDispatch(pack_indices<VEI...> vei, pack_indices<HEI...> hei, pack_indices<AI...> ai, pack_indices<CI...> ci, size_t size, Result &&res, CallArgs&&... args)
			{
			//	assert(this->m_execPlan != NULL && this->m_execPlan->isCalibrated());
			
				size_t Vsize = get_noref<0>(get_noref<VEI>(args...).size()..., this->default_size_j);
				size_t Hsize = get_noref<0>(get_noref<HEI>(args...).size()..., this->default_size_i);
				
				if (disjunction((get<VEI, CallArgs...>(args...).size() < Vsize)...))
					SKEPU_ERROR("Non-matching container sizes");
			
				if (disjunction((get<HEI, CallArgs...>(args...).size() < Hsize)...))
					SKEPU_ERROR("Non-matching container sizes");
				
				this->selectBackend(Vsize + Hsize);
				
				switch (this->m_selected_spec->activateBackend())
				{
				case Backend::Type::Hybrid:
#ifdef SKEPU_HYBRID
					std::cerr << "MapPairsReduce Hybrid: Not implemented" << std::endl;
					//this->Hybrid(Vsize, Hsize, vei, hei, ai, ci, res.begin(), get<VEI, CallArgs...>(args...).begin()..., get<HEI, CallArgs...>(args...).begin()..., get<AI, CallArgs...>(args...)..., get<CI, CallArgs...>(args...)...);
					break;
#endif
				case Backend::Type::CUDA:
#ifdef SKEPU_CUDA
					std::cerr << "MapPairsReduce Hybrid: Not implemented" << std::endl;
					//this->CUDA(0, Vsize, Hsize, vei, hei, ai, ci, res.begin(), get<VEI, CallArgs...>(args...).begin()..., get<HEI, CallArgs...>(args...).begin()..., get<AI, CallArgs...>(args...)..., get<CI, CallArgs...>(args...)...);
					break;
#endif
				case Backend::Type::OpenCL:
#ifdef SKEPU_OPENCL
					this->CL(0, Vsize, Hsize, vei, hei, ai, ci, res.begin(), get<VEI, CallArgs...>(args...).begin()..., get<HEI, CallArgs...>(args...).begin()..., get<AI, CallArgs...>(args...)..., get<CI, CallArgs...>(args...)...);
					break;
#endif
				case Backend::Type::OpenMP:
#ifdef SKEPU_OPENMP
					this->OMP(Vsize, Hsize, vei, hei, ai, ci, res.begin(), get<VEI, CallArgs...>(args...).begin()..., get<HEI, CallArgs...>(args...).begin()..., get<AI, CallArgs...>(args...)..., get<CI, CallArgs...>(args...)...);
					break;
#endif
				default:
					this->CPU(Vsize, Hsize, vei, hei, ai, ci, res.begin(), get<VEI, CallArgs...>(args...).begin()..., get<HEI, CallArgs...>(args...).begin()..., get<AI, CallArgs...>(args...)..., get<CI, CallArgs...>(args...)...);
					break;
				}
			}
			
			
			template<size_t... VEI, size_t... HEI, size_t... AI, size_t... CI, typename Result, typename... CallArgs>
			void CPU(size_t Vsize, size_t Hsize, pack_indices<VEI...> vei, pack_indices<HEI...> hei, pack_indices<AI...>, pack_indices<CI...>, Result &&res, CallArgs&&... args);
			
#ifdef SKEPU_OPENMP
			
			template<size_t... VEI, size_t... HEI, size_t... AI, size_t... CI, typename Result, typename ...CallArgs>
			void OMP(size_t Vsize, size_t Hsize, pack_indices<VEI...> vei, pack_indices<HEI...> hei, pack_indices<AI...>, pack_indices<CI...>, Result &&res, CallArgs&&... args);
			
#endif // SKEPU_OPENMP
			
#ifdef SKEPU_CUDA
			
			template<size_t... VEI, size_t... HEI, size_t... AI, size_t... CI, typename... CallArgs>
			void CUDA(size_t startIdx, size_t Vsize, size_t Hsize, pack_indices<VEI...> vei, pack_indices<HEI...> hei, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args);
			
			template<size_t... VEI, size_t... HEI, size_t... AI, size_t... CI, typename ...CallArgs>
			void mapPairsReduceSingleThread_CU(size_t deviceID, size_t startIdx, size_t Vsize, size_t Hsize, pack_indices<VEI...> vei, pack_indices<HEI...> hei, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args);
			
			template<size_t... VEI, size_t... HEI, size_t... AI, size_t... CI, typename ...CallArgs>
			void mapPairsReduceMultiStream_CU(size_t deviceID, size_t startIdx, size_t Vsize, size_t Hsize, pack_indices<VEI...> vei, pack_indices<HEI...> hei, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args);
			
			template<size_t... VEI, size_t... HEI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapPairsReduceSingleThreadMultiGPU_CU(size_t useNumGPU, size_t startIdx, size_t Vsize, size_t Hsize, pack_indices<VEI...> vei, pack_indices<HEI...> hei, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args);
			
			template<size_t... VEI, size_t... HEI, size_t... AI, size_t... CI, typename ...CallArgs>
			void mapPairsReduceMultiStreamMultiGPU_CU(size_t useNumGPU, size_t startIdx, size_t Vsize, size_t Hsize, pack_indices<VEI...> vei, pack_indices<HEI...> hei, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args);
			
			
#endif // SKEPU_CUDA
			
#ifdef SKEPU_OPENCL
			
			template<size_t... VEI, size_t... HEI, size_t... AI, size_t... CI, typename Result, typename... CallArgs>
			void CL(size_t startIdx, size_t Vsize, size_t Hsize, pack_indices<VEI...> vei, pack_indices<HEI...> hei, pack_indices<AI...>, pack_indices<CI...>, Result &&res, CallArgs&&... args);
			
			template<size_t... VEI, size_t... HEI, size_t... AI, size_t... CI, typename Result, typename... CallArgs>
			void mapPairsReduceNumDevices_CL(size_t numDevices, size_t startIdx, size_t Vsize, size_t Hsize, pack_indices<VEI...> vei, pack_indices<HEI...> hei, pack_indices<AI...>, pack_indices<CI...>, Result &&res, CallArgs&&... args);
			
#endif // SKEPU_OPENCL
		
			
#ifdef SKEPU_HYBRID
			
			template<size_t... VEI, size_t... HEI, size_t... AI, size_t... CI, typename ...CallArgs>
			Ret Hybrid(size_t Vsize, size_t Hsize, pack_indices<VEI...> vei, pack_indices<HEI...> hei, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args);
			
			template<size_t... AI, size_t... CI, typename ...CallArgs>
			Ret Hybrid(size_t Vsize, size_t Hsize, pack_indices<>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args);
			
#endif // SKEPU_HYBRID
			
}; // class MapPairsReduce
		
	} // end namespace backend

#ifdef SKEPU_MERCURIUM

// TODO

#endif // SKEPU_MERCURIUM

} // end namespace skepu


#include "impl/mappairsreduce/mappairsreduce_cpu.inl"
#include "impl/mappairsreduce/mappairsreduce_omp.inl"
#include "impl/mappairsreduce/mappairsreduce_cl.inl"
//#include "impl/mappairsreduce/mappairsreduce_cu.inl"
//#include "impl/mappairsreduce/mappairsreduce_hy.inl"

#endif
