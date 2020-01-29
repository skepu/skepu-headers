#pragma once

#include "skepu3/impl/common.hpp"

namespace skepu
{
	template<int, typename, typename...>
	class MapImpl;
	
	template<int arity = 1, typename Ret, typename... Args>
	MapImpl<arity, Ret, Args...> MapWrapper(std::function<Ret(Args...)> map)
	{
		return MapImpl<arity, Ret, Args...>(map);
	}
	
	// For function pointers
	template<int arity = 1, typename Ret, typename... Args>
	MapImpl<arity, Ret, Args...> Map(Ret(*map)(Args...))
	{
		return MapWrapper<arity>((std::function<Ret(Args...)>)map);
	}
	
	// For lambdas and functors
	template<int arity = 1, typename T>
	auto Map(T map) -> decltype(MapWrapper<arity>(lambda_cast(map)))
	{
		return MapWrapper<arity>(lambda_cast(map));
	}
	
	template<int InArity, typename Ret, typename... Args>
	class MapImpl: public SeqSkeletonBase
	{
		static constexpr bool indexed = is_indexed<Args...>::value;
		static constexpr size_t OutArity = out_size<Ret>::value;
		static constexpr size_t numArgs = sizeof...(Args) - (indexed ? 1 : 0) + OutArity;
		static constexpr size_t anyCont = trait_count_all<is_skepu_container_proxy, Args...>::value;
		
		// Supports the "index trick": using a variant of tag dispatching to index template parameter packs
		static constexpr typename make_pack_indices<OutArity, 0>::type out_indices{};
		static constexpr typename make_pack_indices<InArity + OutArity, OutArity>::type elwise_indices{};
		static constexpr typename make_pack_indices<InArity + OutArity + anyCont, InArity + OutArity>::type any_indices{};
		static constexpr typename make_pack_indices<numArgs, InArity + OutArity + anyCont>::type const_indices{};
		
		using MapFunc = std::function<Ret(Args...)>;
		using F = ConditionalIndexForwarder<indexed, MapFunc>;
		
		// For iterators
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs> 
		void apply(size_t size, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			if (disjunction((get<OI>(args...).size() < size)...))
				SKEPU_ERROR("Non-matching output container sizes");
			
			if (disjunction((get<EI>(args...).size() < size)...))
				SKEPU_ERROR("Non-matching input container sizes");
			
			auto out = std::make_tuple(get<OI>(args...).begin()...);
			auto in  = std::make_tuple(get<EI>(args...).begin()...);
			
			while (size --> 0)
			{
				auto index = std::get<0>(out).getIndex();
				auto res = F::forward(mapFunc, index,
					*std::get<EI-OutArity>(in)++...,
					get<AI>(args...).hostProxy(typename pack_element<AI-OutArity+(indexed ? 1 : 0), typename proxy_tag<Args>::type...>::type{}, index)...,
					get<CI>(args...)...
				);
				std::tie(*std::get<OI>(out)++...) = res;
			}
		}
		
		
	public:
		
		void setDefaultSize(size_t x, size_t y = 0)
		{
			this->default_size_x = x;
			this->default_size_y = y;
		}
		
		template<typename... CallArgs>
		auto operator()(CallArgs&&... args) -> decltype(get<0>(args...))
		{
			static_assert(sizeof...(CallArgs) == numArgs, "Number of arguments not matching Map function");
			this->apply(get<0>(args...).size(), out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
			return get<0>(args...);
		}
		
		template<typename Iterator, typename... CallArgs, REQUIRES_VALUE(is_skepu_iterator<Iterator, Ret>)>
		Iterator operator()(Iterator res, Iterator res_end, CallArgs&&... args)
		{
			static_assert(sizeof...(CallArgs) == numArgs-1, "Number of arguments not matching Map function");
			this->apply(res_end - res, out_indices, elwise_indices, any_indices, const_indices, res, std::forward<CallArgs>(args)...);
			return res;
		}
		
	/*	template<template<class> class Container = Vector, typename... CallArgs>
		Container<Ret> operator()(CallArgs&&... args)
		{
			static_assert(sizeof...(CallArgs) == numArgs, "Number of arguments not matching Map function");
			
			if (this->default_size_y != 0)
			{
				Container<Ret> res(this->default_size_x, this->default_size_y);
				this->apply(elwise_indices, any_indices, const_indices, res, std::forward<CallArgs>(args)...);
				return std::move(res);
			}
			else
			{
				Container<Ret> res(this->default_size_x);
				this->apply(out_indices, elwise_indices, any_indices, const_indices, res.size(), res.begin(), std::forward<CallArgs>(args)...);
				return std::move(res);
		//	}
		}*/
		
	private:
		MapFunc mapFunc;
		MapImpl(MapFunc map): mapFunc(map) {}
		
		size_t default_size_x;
		size_t default_size_y;
		
		friend MapImpl<InArity, Ret, Args...> MapWrapper<InArity, Ret, Args...>(MapFunc);
		
	}; // end class MapImpl
	
} // end namespace skepu
