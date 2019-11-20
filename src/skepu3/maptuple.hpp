#pragma once

#include "skepu3/impl/common.hpp"

namespace skepu
{
	template<typename... args>
	using multiple = std::tuple<args...>;
	
	template <typename... Args>
	auto ret(Args&&... args) -> decltype(std::make_tuple(std::forward<Args>(args)...)) {
		return std::make_tuple(std::forward<Args>(args)...);
	}
	
	
	template<int, typename, typename...>
	class MapTupleImpl;
	
	template<int arity = 1, typename Ret, typename... Args>
	MapTupleImpl<arity, Ret, Args...> MapTupleWrapper(std::function<Ret(Args...)> map)
	{
		return MapTupleImpl<arity, Ret, Args...>(map);
	}
	
	// For function pointers
	template<int arity = 1, typename Ret, typename... Args>
	MapTupleImpl<arity, Ret, Args...> MapTuple(Ret(*map)(Args...))
	{
		return MapTupleWrapper<arity>((std::function<Ret(Args...)>)map);
	}
	
	// For lambdas and functors
	template<int arity = 1, typename T>
	auto MapTuple(T map) -> decltype(MapTupleWrapper<arity>(lambda_cast(map)))
	{
		return MapTupleWrapper<arity>(lambda_cast(map));
	}
	
	template<typename T>
	struct out_size: std::integral_constant<size_t, 1> {};
	
	template<typename... Args>
	struct out_size<std::tuple<Args...>>: std::integral_constant<size_t, sizeof... (Args)> {};
	
	template<int InArity, typename Ret, typename... Args>
	class MapTupleImpl: public SeqSkeletonBase
	{
		static constexpr bool indexed = is_indexed<Args...>::value;
		static constexpr size_t OutArity = out_size<Ret>::value;
		static constexpr size_t numArgs = sizeof...(Args) - (indexed ? 1 : 0) + OutArity;
		static constexpr size_t anyCont = trait_count_all<is_skepu_container_proxy, Args...>::value;
		
	//	using ProxyTypes = std::tuple<typename proxy_tag<Args>::type...>;
		
		// Supports the "index trick": using a variant of tag dispatching to index template parameter packs
		static constexpr typename make_pack_indices<OutArity, 0>::type out_indices{};
		static constexpr typename make_pack_indices<InArity + OutArity, OutArity>::type elwise_indices{};
		static constexpr typename make_pack_indices<InArity + OutArity + anyCont, InArity + OutArity>::type any_indices{};
		static constexpr typename make_pack_indices<numArgs, InArity + OutArity + anyCont>::type const_indices{};
		
		using MapFunc = std::function<Ret(Args...)>;
		using F = ConditionalIndexForwarder<indexed, MapFunc>;
		
		// For iterators
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... Iterator, typename... CallArgs> 
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
		void operator()(CallArgs&&... args)
		{
			static_assert(sizeof...(CallArgs) == numArgs, "Number of arguments not matching Map function");
			this->apply(get<0>(args...).size(), out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
		}
		
	/*	template<typename Iterator, typename... CallArgs, REQUIRES_VALUE(is_skepu_iterator<Iterator, Ret>)>
		Iterator operator()(Iterator res, Iterator res_end, CallArgs&&... args)
		{
			static_assert(sizeof...(CallArgs) == numArgs, "Number of arguments not matching Map function");
			this->apply(elwise_indices, any_indices, const_indices, res_end - res, res, std::forward<CallArgs>(args)...);
			return res;
		}*/
		
	private:
		MapFunc mapFunc;
		MapTupleImpl(MapFunc map): mapFunc(map) {}
		
		size_t default_size_x;
		size_t default_size_y;
		
		friend MapTupleImpl<InArity, Ret, Args...> MapTupleWrapper<InArity, Ret, Args...>(MapFunc);
		
	}; // end class MapTupleImpl
	
	
	
} // end namespace skepu
