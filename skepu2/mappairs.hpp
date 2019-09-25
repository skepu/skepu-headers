#pragma once

#include "skepu2/impl/common.hpp"

namespace skepu2
{
	template<int, int, typename, typename...>
	class MapPairsImpl;
	
	template<int Varity = 1, int Harity = 1, typename Ret, typename... Args>
	MapPairsImpl<Varity, Harity, Ret, Args...> MapPairsWrapper(std::function<Ret(Args...)> mapPairs)
	{
		return MapPairsImpl<Varity, Harity, Ret, Args...>(mapPairs);
	}
	
	// For function pointers
	template<int Varity = 1, int Harity = 1, typename Ret, typename... Args>
	MapPairsImpl<Varity, Harity, Ret, Args...> MapPairs(Ret(*mapPairs)(Args...))
	{
		return MapPairsWrapper<Varity, Harity>((std::function<Ret(Args...)>)mapPairs);
	}
	
	// For lambdas and functors
	template<int Varity = 1, int Harity = 1, typename T>
	auto MapPairs(T mapPairs) -> decltype(MapPairsWrapper<Varity, Harity>(lambda_cast(mapPairs)))
	{
		return MapPairsWrapper<Varity, Harity>(lambda_cast(mapPairs));
	}
	
	template<int Varity, int Harity, typename Ret, typename... Args>
	class MapPairsImpl: public SeqSkeletonBase
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
		using F = ConditionalIndexForwarder<indexed, MapPairsFunc>;
		
		// For iterators
		template<size_t... VEI, size_t... HEI, size_t... AI, size_t... CI, typename Iterator, typename... CallArgs> 
		void apply(pack_indices<VEI...>, pack_indices<HEI...>, pack_indices<AI...>, pack_indices<CI...>, size_t Hsize, size_t Vsize, Iterator res, CallArgs&&... args)
		{
		
		//	std:: cout << "Varity: " << Varity << " Harity: " << Harity << "\n";
		//	std:: cout << "Vsize: " << Vsize << " Hsize: " << Hsize << "\n";
			
			if (disjunction((get<VEI>(args...).size() < Vsize)...))
				SKEPU_ERROR("Non-matching container sizes");
			
			if (disjunction((get<HEI>(args...).size() < Hsize)...))
				SKEPU_ERROR("Non-matching container sizes");
			
			auto HelwiseIterators = std::make_tuple(get<VEI>(args...).begin()...);
			auto VelwiseIterators = std::make_tuple(get<HEI>(args...).begin()...);
			
			for (size_t i = 0; i < Vsize; ++i)
			{
				for (size_t j = 0; j < Hsize; ++j)
				{
					auto index = Index2D { i, j };
					res(i, j) = F::forward(mapPairsFunc, index, std::get<VEI>(HelwiseIterators)(i)..., std::get<HEI-Varity>(VelwiseIterators)(j)..., get<AI>(args...).hostProxy()..., get<CI>(args...)...);
				}
			}
			
		}
		
		
	public:
		
		void setDefaultSize(size_t x, size_t y = 0)
		{
			this->default_size_x = x;
			this->default_size_y = y;
		}
		
		template<template<class> class Container, typename... CallArgs>
		Container<Ret> &operator()(Container<Ret> &res, CallArgs&&... args)
		{
			static_assert(sizeof...(CallArgs) == numArgs, "Number of arguments not matching Map function");
			this->apply(Helwise_indices, Velwise_indices, any_indices, const_indices, res.total_cols(), res.total_rows(), res.begin(), std::forward<CallArgs>(args)...);
			return res;
		}
		
		template<typename Iterator, typename... CallArgs, REQUIRES(is_skepu_iterator<Iterator, Ret>())>
		Iterator operator()(Iterator res, Iterator res_end, CallArgs&&... args)
		{
			static_assert(sizeof...(CallArgs) == numArgs, "Number of arguments not matching Map function");
			this->apply(Helwise_indices, Velwise_indices, any_indices, const_indices, res_end - res, res, std::forward<CallArgs>(args)...);
			return res;
		}
		
		template<template<class> class Container = Vector, typename... CallArgs>
		Container<Ret> operator()(CallArgs&&... args)
		{
			static_assert(sizeof...(CallArgs) == numArgs, "Number of arguments not matching Map function");
			
		/*	if (this->default_size_y != 0)
			{
				Container<Ret> res(this->default_size_x, this->default_size_y);
				this->apply(elwise_indices, any_indices, const_indices, res, std::forward<CallArgs>(args)...);
				return std::move(res);
			}
			else
			{*/
				Container<Ret> res(this->default_size_x);
				this->apply(Helwise_indices, Velwise_indices, any_indices, const_indices, res.size(), res.begin(), std::forward<CallArgs>(args)...);
				return std::move(res);
		//	}
		}
		
	private:
		MapPairsFunc mapPairsFunc;
		MapPairsImpl(MapPairsFunc mapPairs): mapPairsFunc(mapPairs) {}
		
		size_t default_size_x = 1;
		size_t default_size_y = 1;
		
		friend MapPairsImpl<Varity, Harity, Ret, Args...> MapPairsWrapper<Varity, Harity, Ret, Args...>(MapPairsFunc);
		
	}; // end class MapImpl
	
} // end namespace skepu2
