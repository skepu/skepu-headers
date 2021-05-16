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
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename ...CallArgs, REQUIRES_B(!MapFunc::usesPRNG && true && sizeof...(CallArgs) >= 0)>
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
			for (Ret const& parsum : parsums)
				pack_expand((get_or_return<OI>(res) = ReduceFunc::OMP(get_or_return<OI>(res), get_or_return<OI>(parsum)), 0)...);
			
			return res;
		}
		
		
		
		template<size_t arity, typename MapFunc, typename ReduceFunc, typename CUDAKernel, typename CUDAReduceKernel, typename CLKernel>
		template<size_t... OI, size_t... AI, size_t... CI, typename ...CallArgs, REQUIRES_B(!MapFunc::usesPRNG && true && sizeof...(CallArgs) >= 0)>
		typename MapFunc::Ret MapReduce<arity, MapFunc, ReduceFunc, CUDAKernel, CUDAReduceKernel, CLKernel>
		::OMP(size_t size, pack_indices<OI...>, pack_indices<>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args)
		{
			DEBUG_TEXT_LEVEL1("OpenMP MapReduce: size = " << size);
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
			for (Ret const& parsum : parsums)
				pack_expand((get_or_return<OI>(res) = ReduceFunc::OMP(get_or_return<OI>(res), get_or_return<OI>(parsum)), 0)...);
			
			return res;
		}
		
		
		
		
		// PRNG
		
		
		template<size_t arity, typename MapFunc, typename ReduceFunc, typename CUDAKernel, typename CUDAReduceKernel, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename ...CallArgs, REQUIRES_B(!!MapFunc::usesPRNG && sizeof...(CallArgs) >= 0)>
		typename MapFunc::Ret MapReduce<arity, MapFunc, ReduceFunc, CUDAKernel, CUDAReduceKernel, CLKernel>
		::OMP(size_t size, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args)
		{
			// Sync with device data
			pack_expand((get<EI, CallArgs...>(args...).getParent().updateHost(), 0)...);
			pack_expand((get<AI, CallArgs...>(args...).getParent().updateHost(hasReadAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);
			pack_expand((get<AI, CallArgs...>(args...).getParent().invalidateDeviceData(hasWriteAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);
			
			if (this->m_prng == nullptr)
				SKEPU_ERROR("No random stream set in skeleton instance");

			size_t threads = omp_get_max_threads();
			auto randoms = this->m_prng->template asRandom<MapFunc::randomCount>(size, threads);
			
			std::vector<Ret> parsums(std::min<size_t>(size, omp_get_max_threads()));
			bool first = true;
			
			// Perform Map and partial Reduce with OpenMP
#pragma omp parallel for schedule(runtime) firstprivate(first)
			for (size_t i = 0; i < size; ++i)
			{
				size_t myid = omp_get_thread_num();
				auto &my_random = randoms(myid);
				auto index = (get<0, CallArgs...>(args...) + i).getIndex();
				Ret tempMap = F::forward(MapFunc::OMP,
					index, my_random,
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
			for (Ret const& parsum : parsums)
				pack_expand((get_or_return<OI>(res) = ReduceFunc::OMP(get_or_return<OI>(res), get_or_return<OI>(parsum)), 0)...);
			
			return res;
		}
		
		
		
		template<size_t arity, typename MapFunc, typename ReduceFunc, typename CUDAKernel, typename CUDAReduceKernel, typename CLKernel>
		template<size_t... OI, size_t... AI, size_t... CI, typename ...CallArgs, REQUIRES_B(!!MapFunc::usesPRNG && sizeof...(CallArgs) >= 0)>
		typename MapFunc::Ret MapReduce<arity, MapFunc, ReduceFunc, CUDAKernel, CUDAReduceKernel, CLKernel>
		::OMP(size_t size, pack_indices<OI...>, pack_indices<>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args)
		{
			DEBUG_TEXT_LEVEL1("OpenMP MapReduce: size = " << size);
			// Sync with device data
			pack_expand((get<AI, CallArgs...>(args...).getParent().updateHost(hasReadAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);
			pack_expand((get<AI, CallArgs...>(args...).getParent().invalidateDeviceData(hasWriteAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);
			
			if (this->m_prng == nullptr)
				SKEPU_ERROR("No random stream set in skeleton instance");

			size_t threads = omp_get_max_threads();
			auto randoms = this->m_prng->template asRandom<MapFunc::randomCount>(size, threads);
			
			std::vector<Ret> parsums(std::min<size_t>(size, omp_get_max_threads()));
			bool first = true;
			
#ifdef SKEPU_PRNG_VERIFY_FORWARD
			for (auto &r : randoms)
			{
				std::cout << "State: " << r.m_state << " valid uses: " << r.m_valid_uses << "\n";
			}
#endif
			
			// Perform Map and partial Reduce with OpenMP
#pragma omp parallel for schedule(runtime) firstprivate(first)
			for (size_t i = 0; i < size; ++i)
			{
				size_t myid = omp_get_thread_num();
				auto &my_random = randoms(myid);
				auto index = make_index(defaultDim{}, i, this->default_size_j, this->default_size_k, this->default_size_l);
				Ret tempMap = F::forward(MapFunc::OMP,
					index, my_random,
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
			for (Ret const& parsum : parsums)
				pack_expand((get_or_return<OI>(res) = ReduceFunc::OMP(get_or_return<OI>(res), get_or_return<OI>(parsum)), 0)...);
			
			return res;
		}
		
	} // namespace backend
} // namespace skepu

#endif
