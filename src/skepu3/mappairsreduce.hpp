#pragma once

#include "skepu3/impl/common.hpp"

namespace skepu
{
	template<int, int, typename, typename...>
	class MapPairsReduceImpl;
	
	template<int Varity = 1, int Harity = 1, typename Ret, typename... Args>
	MapPairsReduceImpl<Varity, Harity, Ret, Args...> MapPairsReduceWrapper(std::function<Ret(Args...)> mapPairs, std::function<Ret(Ret, Ret)> red)
	{
		return MapPairsReduceImpl<Varity, Harity, Ret, Args...>(mapPairs, red);
	}
	
	// For function pointers
	template<int Varity = 1, int Harity = 1, typename Ret, typename... Args>
	MapPairsReduceImpl<Varity, Harity, Ret, Args...> MapPairsReduce(Ret(*mapPairs)(Args...), Ret(*red)(Ret, Ret))
	{
		return MapPairsReduceWrapper<Varity, Harity>((std::function<Ret(Args...)>)mapPairs, (std::function<Ret(Ret, Ret)>)red);
	}
	
	// For lambdas and functors
	template<int Varity = 1, int Harity = 1, typename T1, typename T2>
	auto MapPairsReduce(T1 mapPairs, T2 red) -> decltype(MapPairsReduceWrapper<Varity, Harity>(lambda_cast(mapPairs), lambda_cast(red)))
	{
		return MapPairsReduceWrapper<Varity, Harity>(lambda_cast(mapPairs), lambda_cast(red));
	}
	
	template<int Varity, int Harity, typename Ret, typename... Args>
	class MapPairsReduceImpl: public SeqSkeletonBase
	{
		static constexpr bool indexed = is_indexed<Args...>::value;
		static constexpr size_t numArgs = sizeof...(Args) - (indexed ? 1 : 0);
		static constexpr size_t anyCont = trait_count_all<is_skepu_container_proxy, Args...>::value;
		
		// Supports the "index trick": using a variant of tag dispatching to index template parameter packs
		static constexpr typename make_pack_indices<Varity, 0>::type Helwise_indices{};
		static constexpr typename make_pack_indices<Harity + Varity, Varity>::type Velwise_indices{};
		static constexpr typename make_pack_indices<Varity + Harity + anyCont, Varity + Harity>::type any_indices{};
		static constexpr typename make_pack_indices<numArgs, Varity + Harity + anyCont>::type const_indices{};
		
		using MapPairsFunc = std::function<Ret(Args...)>;
		using RedFunc = std::function<Ret(Ret, Ret)>;
		using F = ConditionalIndexForwarder<indexed, MapPairsFunc>;
		
		// For iterators
		template<size_t... VEI, size_t... HEI, size_t... AI, size_t... CI, typename Iterator, typename... CallArgs>
		void apply(pack_indices<VEI...>, pack_indices<HEI...>, pack_indices<AI...>, pack_indices<CI...>, size_t size, Iterator res, CallArgs&&... args)
		{
			size_t Vsize = get_noref<0>(get_noref<VEI>(args...).size()..., this->default_size_y);
			size_t Hsize = get_noref<0>(get_noref<HEI>(args...).size()..., this->default_size_x);
			
			if (disjunction((get<VEI>(args...).size() < Vsize)...))
				SKEPU_ERROR("Non-matching input container sizes");
			
			if (disjunction((get<HEI>(args...).size() < Hsize)...))
				SKEPU_ERROR("Non-matching input container sizes");
			
			if ((this->m_mode == ReduceMode::RowWise && size < Vsize) || (this->m_mode == ReduceMode::ColWise && size < Hsize))
				SKEPU_ERROR("Non-matching output container size");
			
			auto HelwiseIterators = std::make_tuple(get<VEI>(args...).begin()...);
			auto VelwiseIterators = std::make_tuple(get<HEI>(args...).begin()...);
			
			if (this->m_mode == ReduceMode::RowWise)
				for (size_t i = 0; i < Vsize; ++i)
				{
					res(i) = this->m_start;
					for (size_t j = 0; j < Hsize; ++j)
					{
						auto index = Index2D { i, j };
						Ret temp = F::forward(mapPairsFunc, index, std::get<VEI>(HelwiseIterators)(i)..., std::get<HEI-Varity>(VelwiseIterators)(j)..., get<AI>(args...).hostProxy()..., get<CI>(args...)...);
						res(i) = redFunc(res(i), temp);
					}
				}
			else if (this->m_mode == ReduceMode::ColWise)
				for (size_t j = 0; j < Hsize; ++j)
				{
					res(j) = this->m_start;
					for (size_t i = 0; i < Vsize; ++i)
					{
						auto index = Index2D { i, j };
						Ret temp = F::forward(mapPairsFunc, index, std::get<VEI>(HelwiseIterators)(i)..., std::get<HEI-Varity>(VelwiseIterators)(j)..., get<AI>(args...).hostProxy()..., get<CI>(args...)...);
						res(j) = redFunc(res(j), temp);
					}
				}
			
		}
		
		
	public:
		
		void setStartValue(Ret val)
		{
			this->m_start = val;
		}
		
		void setReduceMode(ReduceMode mode)
		{
			this->m_mode = mode;
		}
		
		void setDefaultSize(size_t x, size_t y = 0)
		{
			this->default_size_x = x;
			this->default_size_y = y;
		}
		
		template<typename... CallArgs>
		Vector<Ret> &operator()(Vector<Ret> &res, CallArgs&&... args)
		{
			static_assert(sizeof...(CallArgs) == numArgs, "Number of arguments not matching Map function");
			this->apply(Helwise_indices, Velwise_indices, any_indices, const_indices, res.size(), res.begin(), std::forward<CallArgs>(args)...);
			return res;
		}
		
		template<typename Iterator, typename... CallArgs, REQUIRES(is_skepu_iterator<Iterator, Ret>())>
		Iterator operator()(Iterator res, Iterator res_end, CallArgs&&... args)
		{
			static_assert(sizeof...(CallArgs) == numArgs, "Number of arguments not matching Map function");
			this->apply(Helwise_indices, Velwise_indices, any_indices, const_indices, res_end - res, res, std::forward<CallArgs>(args)...);
			return res;
		}
		
	private:
		MapPairsFunc mapPairsFunc;
		RedFunc redFunc;
		MapPairsReduceImpl(MapPairsFunc mapPairs, RedFunc red): mapPairsFunc(mapPairs), redFunc(red) {}
		
		ReduceMode m_mode = ReduceMode::RowWise;
		Ret m_start{};
		size_t default_size_x = 1;
		size_t default_size_y = 1;
		
		friend MapPairsReduceImpl<Varity, Harity, Ret, Args...> MapPairsReduceWrapper<Varity, Harity, Ret, Args...>(MapPairsFunc, RedFunc);
		
	}; // end class MapImpl
	
} // end namespace skepu
