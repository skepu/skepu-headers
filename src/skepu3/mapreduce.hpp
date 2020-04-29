#pragma once

#include <utility>

#include "skepu3/impl/common.hpp"

namespace skepu
{
	template<int, typename, typename...>
	class MapReduceImpl;
	
	template<int arity = 1, typename Ret, typename... Args>
	MapReduceImpl<arity, Ret, Args...> MapReduceWrapper(std::function<Ret(Args...)> map, std::function<Ret(Ret, Ret)> red)
	{
		return MapReduceImpl<arity, Ret, Args...>(map, red);
	}
	
	// For function pointers
	template<int arity = 1, typename Ret, typename... Args>
	MapReduceImpl<arity, Ret, Args...> MapReduce(Ret(*map)(Args...), Ret(*red)(Ret, Ret))
	{
		return MapReduceWrapper<arity>((std::function<Ret(Args...)>)map, (std::function<Ret(Ret, Ret)>)red);
	}
	
	// For lambdas and functors
	template<int arity = 1, typename T1, typename T2>
	auto MapReduce(T1 map, T2 red) -> decltype(MapReduceWrapper<arity>(lambda_cast(map), lambda_cast(red)))
	{
		return MapReduceWrapper<arity>(lambda_cast(map), lambda_cast(red));
	}
	
	
	/* MapReduce "semantic guide" for the SkePU 2 precompiler.
	 * Sequential implementation when used with any C++ compiler.
	 * Works with any number of variable arguments > 0.
	 * Works with any number of constant arguments >= 0.
	 */
	template<int arity, typename Ret, typename... Args>
	class MapReduceImpl: public SeqSkeletonBase
	{
		using MapFunc = std::function<Ret(Args...)>;
		using RedFunc = std::function<Ret(Ret, Ret)>;
		
		static constexpr bool indexed = is_indexed<Args...>::value;
		static constexpr size_t numArgs = sizeof...(Args) - (indexed ? 1 : 0);
		static constexpr size_t anyCont = trait_count_all<is_skepu_container_proxy, Args...>::value;
		
		using defaultDim = typename std::conditional<indexed, index_dimension<typename first_element<Args...>::type>, std::integral_constant<int, 1>>::type;
		using First = typename pack_element<indexed ? 1 : 0, Args...>::type;
		
		// Supports the "index trick": using a variant of tag dispatching to index template parameter packs
		static constexpr typename make_pack_indices<arity, 0>::type elwise_indices{};
		static constexpr typename make_pack_indices<arity + anyCont, arity>::type any_indices{};
		static constexpr typename make_pack_indices<numArgs, arity + anyCont>::type const_indices{};
		
		using F = ConditionalIndexForwarder<indexed, MapFunc>;
		
		
		template<size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		Ret apply(pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, size_t size, CallArgs&&... args)
		{
			if (disjunction((get<EI>(args...).size() < size)...))
				SKEPU_ERROR("MapReduce: Non-matching container sizes");
			
			Ret res = this->m_start;
			auto elwiseIterators = std::make_tuple(get<EI>(args...).begin()...);
			
			while (size --> 0)
			{
				auto index = std::get<0>(elwiseIterators).getIndex();
				Ret temp = F::forward(mapFunc, index, *std::get<EI>(elwiseIterators)++..., get<AI>(args...).hostProxy()..., get<CI>(args...)...);
				res = redFunc(res, temp);
			}
			
			return res;
		}
		
		template<size_t... AI, size_t... CI, typename... CallArgs>
		Ret zero_apply(pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			size_t size = this->default_size_i;
			if (defaultDim::value >= 2) size *= this->default_size_j;
			if (defaultDim::value >= 3) size *= this->default_size_k;
			if (defaultDim::value >= 4) size *= this->default_size_l;
			
			Ret res = this->m_start;
			for (size_t i = 0; i < size; ++i)
			{
				Ret temp = F::forward(mapFunc,
					make_index(defaultDim{}, i, this->default_size_j, this->default_size_k, this->default_size_l),
					get<AI>(args...).hostProxy()..., get<CI>(args...)...
				);
				res = redFunc(res, temp);
			}
			return res;
		}
		
		
	public:
		
		void setStartValue(Ret val)
		{
			this->m_start = val;
		}
		
		void setDefaultSize(size_t i, size_t j = 0, size_t k = 0, size_t l = 0)
		{
			this->default_size_i = i;
			this->default_size_j = j;
			this->default_size_k = k;
			this->default_size_l = l;
		}
		
		template<template<class> class Container, typename... CallArgs, REQUIRES_VALUE(is_skepu_container<Container<First>>)>
		Ret operator()(Container<First>& arg1, CallArgs&&... args)
		{
			static_assert(sizeof...(CallArgs) + 1 == numArgs, "Number of arguments not matching Map function");
			return apply(elwise_indices, any_indices, const_indices, arg1.size(), arg1.begin(), std::forward<CallArgs>(args)...);
		}
		
		template<template<class> class Container, typename... CallArgs, REQUIRES_VALUE(is_skepu_container<Container<First>>)>
		Ret operator()(const Container<First>& arg1, CallArgs&&... args)
		{
			static_assert(sizeof...(CallArgs) + 1 == numArgs, "Number of arguments not matching Map function");
			return apply(elwise_indices, any_indices, const_indices, arg1.size(), arg1.begin(), std::forward<CallArgs>(args)...);
		}
		
		
		template<typename Iterator, typename... CallArgs, REQUIRES_VALUE(is_skepu_iterator<Iterator, First>)>
		Ret operator()(Iterator arg1, Iterator arg1_end, CallArgs&&... args)
		{
			static_assert(sizeof...(CallArgs) + 1 == numArgs, "Number of arguments not matching Map function");
			return apply(elwise_indices, any_indices, const_indices, arg1_end - arg1, arg1, std::forward<CallArgs>(args)...);
		}
		
		template<template<class> class Container = Vector, typename... CallArgs>
		Ret operator()(CallArgs&&... args)
		{
			static_assert(sizeof...(CallArgs) == numArgs, "Number of arguments not matching Map function");
			return this->zero_apply(any_indices, const_indices, std::forward<CallArgs>(args)...);
		}
		
	private:
		MapFunc mapFunc;
		RedFunc redFunc;
		MapReduceImpl(MapFunc map, RedFunc red): mapFunc(map), redFunc(red) {}
		
		Ret m_start{};
		size_t default_size_i = 0;
		size_t default_size_j = 0;
		size_t default_size_k = 0;
		size_t default_size_l = 0;
		
		friend MapReduceImpl<arity, Ret, Args...> MapReduceWrapper<arity, Ret, Args...>(MapFunc, RedFunc);
		
	}; // end class MapReduce
	
} // end namespace skepu
