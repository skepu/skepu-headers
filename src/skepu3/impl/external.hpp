#pragma once
#ifndef SKEPU_EXTERNAL_HPP
#define SKEPU_EXTERNAL_HPP 1

#include "common.hpp"
#include "meta_helpers.hpp"

namespace skepu {
template<
	size_t ... CI,
	typename OP,
	typename ... Containers,
	REQUIRES_VALUE(are_skepu_containers<Containers...>)>
auto inline
external_impl(
	pack_indices<CI...>,
	OP && op,
	Containers && ... containers) noexcept
-> void
{
	pack_expand((get<CI>(containers...).flush(),0)...);
	op();
}

template<
	size_t ... CI,
	typename OP,
	typename ... Args>
auto inline
external_fwd(
	pack_indices<CI...> ci,
	OP && op,
	Args && ... args) noexcept
-> void
{
	external_impl(
		ci,
		op,
		std::forward<decltype(get<CI>(args...))>(get<CI>(args...))...);
}

/** Execute some code with containers synchronized.
 *
 * \tparam Args A pack containing the types of the SkePU containers and an
					 operator as the last type in the pack.
 *
 * \param args A list of containers and an operator as the last argument.
 */
template<typename ... Args>
auto inline
external(Args && ... args) noexcept
-> void
{
	auto constexpr last_index = sizeof...(Args) -1;
	auto constexpr container_indices =
		typename make_pack_indices<last_index>::type{};

	external_fwd(
		container_indices,
		get<last_index, Args...>(args...),
		std::forward<Args>(args)...);
}

} // namespace skepu

#endif // SKEPU_EXTERNAL_HPP
