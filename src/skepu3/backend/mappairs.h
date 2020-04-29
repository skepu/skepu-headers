/*! \file map.h
 *  \brief Contains a class declaration for the MapPairs skeleton.
 */

#ifndef MAPPAIRS_H
#define MAPPAIRS_H

namespace skepu
{
	namespace backend
	{
		/*!
		 *  \ingroup skeletons
 		 */
		/*!
		 *  \class MapPairs
		 *
		 *  \brief A class representing the MapPairs skeleton.
		 *
		 *  This class defines the MapPairs skeleton, a calculation pattern where a user function is applied to each element of the cartesian product of input vectors.
		 *  Result is a Matrix.
		 */
		template<size_t Varity, size_t Harity, typename MapPairsFunc, typename CUDAKernel, typename CLKernel>
		class MapPairs : public SkeletonBase
		{
			// ==========================    Type definitions   ==========================
			
			using T = typename MapPairsFunc::Ret;
			using F = ConditionalIndexForwarder<MapPairsFunc::indexed, decltype(&MapPairsFunc::CPU)>;
			
			// ==========================     Class members     ==========================
			
			static constexpr size_t numArgs = MapPairsFunc::totalArity - (MapPairsFunc::indexed ? 1 : 0);
			static constexpr size_t anyArity = std::tuple_size<typename MapPairsFunc::ContainerArgs>::value;
			
			// ==========================    Instance members   ==========================
			
			CUDAKernel m_cuda_kernel;
			
			size_t default_size_x;
			size_t default_size_y;
			
		public:
			
			static constexpr auto skeletonType = SkeletonType::MapPairs;
			using ResultArg = std::tuple<T>;
			using ElwiseArgs = typename MapPairsFunc::ElwiseArgs;
			using ContainerArgs = typename MapPairsFunc::ContainerArgs;
			using UniformArgs = typename MapPairsFunc::UniformArgs;
			
			static constexpr typename make_pack_indices<Varity, 0>::type Velwise_indices{};
			static constexpr typename make_pack_indices<Harity + Varity, Varity>::type Helwise_indices{};
			static constexpr typename make_pack_indices<Varity + Harity + anyArity, Varity + Harity>::type any_indices{};
			static constexpr typename make_pack_indices<numArgs, Varity + Harity + anyArity>::type const_indices{};
			
			// =========================      Constructors      ==========================
			
			MapPairs(CUDAKernel kernel) : m_cuda_kernel(kernel)
			{
#ifdef SKEPU_OPENCL
				CLKernel::initialize();
#endif
			}
			
			// =======================  Persistent parameters   ==========================
			
			void setDefaultSize(size_t x, size_t y = 0)
			{
				this->default_size_x = x;
				this->default_size_y = y;
			}
			
			template<typename... Args>
			void tune(Args&&... args)
			{
				tuner::tune(*this, std::forward<Args>(args)...);
			}
			
			// =======================      Call operators      ==========================
			
			template<template<class> class Container, typename... CallArgs, REQUIRES(is_skepu_container<Container<T>>::value)>
			Container<T> &operator()(Container<T> &res, CallArgs&&... args)
			{
				this->backendDispatch(Velwise_indices, Helwise_indices, any_indices, const_indices, res.total_cols(), res.total_rows(), res.begin(), args...);
				return res;
			}
			
		private:
			
			// ==========================    Implementation     ==========================
			
			template<size_t... VEI, size_t... HEI,  size_t... AI, size_t... CI, typename Iterator, typename... CallArgs>
			void CPU(size_t Vsize, size_t Hsize,  pack_indices<VEI...>, pack_indices<HEI...>, pack_indices<AI...>, pack_indices<CI...>, Iterator res, CallArgs&&... args);
			
			
#ifdef SKEPU_OPENMP
			
			template<size_t... VEI, size_t... HEI, size_t... AI, size_t... CI, typename Iterator, typename ...CallArgs>
			void OMP(size_t Vsize, size_t Hsize,  pack_indices<VEI...>, pack_indices<HEI...>, pack_indices<AI...>, pack_indices<CI...>, Iterator res, CallArgs&&... args);
			
#endif // SKEPU_OPENMP
			
			
#ifdef SKEPU_CUDA
			
			template<size_t... VEI, size_t... HEI, size_t... AI, size_t... CI, typename Iterator, typename... CallArgs>
			void CUDA(size_t startIdx, size_t Vsize, size_t Hsize, pack_indices<VEI...>, pack_indices<HEI...>, pack_indices<AI...>, pack_indices<CI...>, Iterator res, CallArgs&&... args);
			
#endif // SKEPU_CUDA
			
			
#ifdef SKEPU_OPENCL
			
			template<size_t... VEI, size_t... HEI, size_t... AI, size_t... CI, typename Iterator, typename... CallArgs>
			void CL(size_t startIdx, size_t Vsize, size_t Hsize, pack_indices<VEI...>, pack_indices<HEI...>, pack_indices<AI...>, pack_indices<CI...>, Iterator res, CallArgs&&... args);
			
			template<size_t... VEI, size_t... HEI, size_t... AI, size_t... CI, typename Iterator, typename... CallArgs>
			void mapNumDevices_CL(size_t startIdx, size_t numDevices, size_t Vsize, size_t Hsize, pack_indices<VEI...>, pack_indices<HEI...>, pack_indices<AI...>, pack_indices<CI...>, Iterator res, CallArgs&&... args);
			
#endif // SKEPU_OPENCL
  
#ifdef SKEPU_HYBRID
			
			template<size_t... VEI, size_t... HEI, size_t... AI, size_t... CI, typename Iterator, typename... CallArgs>
			void Hybrid(size_t Vsize, size_t Hsize, pack_indices<VEI...>, pack_indices<HEI...>, pack_indices<AI...>, pack_indices<CI...>, Iterator res, CallArgs&&... args);
			
#endif // SKEPU_HYBRID
			
			template<size_t... VEI, size_t... HEI, size_t... AI, size_t... CI, typename Iterator, typename... CallArgs>
			void backendDispatch(pack_indices<VEI...> vei, pack_indices<HEI...> hei, pack_indices<AI...> ai, pack_indices<CI...> ci, size_t Hsize, size_t Vsize, Iterator res, CallArgs&&... args)
			{
			//	assert(this->m_execPlan != nullptr && this->m_execPlan->isCalibrated());
				
				if (disjunction((get<VEI, CallArgs...>(args...).size() < Vsize)...))
					SKEPU_ERROR("Non-matching container sizes");
				
				if (disjunction((get<HEI, CallArgs...>(args...).size() < Hsize)...))
					SKEPU_ERROR("Non-matching container sizes");
				
				this->selectBackend(Vsize + Hsize);
				
				switch (this->m_selected_spec->activateBackend())
				{
				case Backend::Type::Hybrid:
#ifdef SKEPU_HYBRID
					std::cerr << "MapPairs Hybrid: Not implemented" << std::endl;
				//	this->Hybrid(size, vei, hei, ai, ci, res, get<EI, CallArgs...>(args...).begin()..., get<AI, CallArgs...>(args...)..., get<CI, CallArgs...>(args...)...);
					break;
#endif
				case Backend::Type::CUDA:
#ifdef SKEPU_CUDA
					std::cerr << "MapPairs CUDA: Not implemented" << std::endl;
				//	this->CUDA(0, size, vei, hei, ai, ci, res, get<EI, CallArgs...>(args...).begin()..., get<AI, CallArgs...>(args...)..., get<CI, CallArgs...>(args...)...);
					break;
#endif
				case Backend::Type::OpenCL:
#ifdef SKEPU_OPENCL
					this->CL(0, Vsize, Hsize, vei, hei, ai, ci, res, get<VEI, CallArgs...>(args...).begin()..., get<HEI, CallArgs...>(args...).begin()..., get<AI, CallArgs...>(args...)..., get<CI, CallArgs...>(args...)...);
					break;
#endif
				case Backend::Type::OpenMP:
#ifdef SKEPU_OPENMP
					this->OMP(Vsize, Hsize, vei, hei, ai, ci, res, get<VEI, CallArgs...>(args...).begin()..., get<HEI, CallArgs...>(args...).begin()..., get<AI, CallArgs...>(args...)..., get<CI, CallArgs...>(args...)...);
					break;
#endif
				default:
					this->CPU(Vsize, Hsize, vei, hei, ai, ci, res, get<VEI, CallArgs...>(args...).begin()..., get<HEI, CallArgs...>(args...).begin()...,get<AI, CallArgs...>(args...)..., get<CI, CallArgs...>(args...)...);
					break;
				}
			}
		
		}; // class MapPairs
		
	} // namespace backend

#ifdef SKEPU_MERCURIUM
template<typename Ret, typename... Args>
struct MapPairsImpl : public SeqSkeletonBase
{
	MapPairsImpl(Ret(*)(Args...));

	void setDefaultSize(size_t x, size_t y = 0);

	template<template<class> class Container, typename... CallArgs>
	Container<Ret> &operator()(Container<Ret> &res, CallArgs&&... args);

	template<
		typename Iterator,
		typename... CallArgs,
		REQUIRES_VALUE(is_skepu_iterator<Iterator, Ret>)>
	Iterator operator()(Iterator res, Iterator res_end, CallArgs&&... args);

	template<template<class> class Container = Vector, typename... CallArgs>
	Container<Ret> operator()(CallArgs&&... args);
};

template<int Varity, int Harity, typename Ret, typename... Args>
auto inline
MapPairs(Ret(*)(Args...))
-> MapPairsImpl<Ret, Args...>;
#endif // SKEPU_MERCURIUM

} // namespace skepu


#include "impl/mappairs/mappairs_cpu.inl"
#include "impl/mappairs/mappairs_omp.inl"
#include "impl/mappairs/mappairs_cl.inl"
//#include "impl/mappairs/mappairs_cu.inl"
//#include "impl/mappairs/mappairs_hy.inl"

#endif // MAPPAIRS_H
