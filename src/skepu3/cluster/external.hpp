#pragma once
#ifndef SKEPU_CLUSTER_EXTERNAL_HPP
#define SKEPU_CLUSTER_EXTERNAL_HPP 1

#include "cluster.hpp"
#include "common.hpp"

namespace skepu {
namespace {

template<
	size_t ... CI,
	typename OP,
	typename ... Containers,
	REQUIRES_VALUE(are_skepu_containers<Containers...>)>
auto inline
external_impl(
	pack_indices<CI...>,
	OP && op,
	Containers && ... args) noexcept
-> void
{
	pack_expand((cont::getParent(get<CI>(args...)).gather_to_root(),0)...);

	if(!cluster::mpi_rank())
		op();

	pack_expand((cont::getParent(get<CI>(args...)).scatter_from_root(),0)...);
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

} // unnamed namespace

template<typename ... Args>
auto inline
external(Args && ... args) noexcept
-> void
{
	size_t static constexpr last_index = sizeof...(Args) -1;
	auto static constexpr container_indices =
		typename make_pack_indices<last_index>::type{};

	external_fwd(
		container_indices,
		get<last_index, Args...>(args...),
		std::forward<Args>(args)...);
}

} // namespace skepu

#endif // SKEPU_CLUSTER_EXTERNAL_HPP
