#pragma once
#ifndef SKEPU_EXTERNAL_HPP
#define SKEPU_EXTERNAL_HPP 1

#include "common.hpp"
#include "meta_helpers.hpp"

namespace skepu {
namespace label {

/** A label to mark that a tuple contains SkePU containers that will be read in
 *	in an skepu::external call.
 */
struct read {};

/** A label to mark that a tuple contains SkePU containers that will be written
 * to in an skepu::external call.
 */
struct write {};

} // namespace label

/** Specify a list of containers that will be read during an external call. */
template<
	typename ... Containers,
	REQUIRES_VALUE(are_skepu_containers<Containers...>)>
auto inline constexpr
read(Containers && ... cs) noexcept
-> std::tuple<label::read, Containers...>
{
	return std::tuple<label::read, Containers...>(label::read{}, cs...);
}

/** Specify a list of continaers that will be written to during an external
 * call */
template<
	typename ... Containers,
	REQUIRES_VALUE(are_skepu_containers<Containers...>)>
auto inline constexpr
write(
	Containers && ... cs) noexcept
-> std::tuple<label::write, Containers...>
{
	return std::tuple<label::write, Containers...>(label::write{}, cs...);
}

/** Run an operation which under no circumstance can be executed in prarallel.
 *
 * \tparam ContR	The types of the readable containers.
 * \tparam OP			The operator.
 * \tparam ContW	The types of the writeable containers.
 *
 * \param cr	The list of readable containers.
 * \param op	The operator to call.
 * \param cw	The list of writeable containers.
 */
template<
	typename ... ContR,
	typename OP,
	typename ... ContW>
auto inline
external(
	std::tuple<label::read, ContR...> cr,
	OP && op,
	std::tuple<label::write, ContW...> cw) noexcept
-> void;

template<
	size_t ... CI,
	typename ... Containers>
auto inline
prepare_read(
	pack_indices<CI...>,
	std::tuple<Containers...> & cs) noexcept
-> void
{
	pack_expand((std::get<CI>(cs).flush(),0)...);
}

template<typename OP>
auto inline
external(OP && op) noexcept
-> void
{
	op();
}

template<
	typename ... ContR,
	typename OP>
auto inline
external(
	std::tuple<label::read, ContR...> cr,
	OP && op) noexcept
-> void
{
	auto constexpr read_indices =
		typename make_pack_indices<sizeof...(ContR) +1, 1>::type{};

	prepare_read(read_indices, cr);

	op();
}

template<
	typename OP,
	typename ... ContW>
auto inline
external(
	OP && op,
	std::tuple<label::write, ContW...>) noexcept
-> void
{
	op();
}

template<
	typename ... ContR,
	typename OP,
	typename ... ContW>
auto inline
external(
	std::tuple<label::read, ContR...> cr,
	OP && op,
	std::tuple<label::write, ContW...>) noexcept
-> void
{
	auto constexpr read_indices =
		typename make_pack_indices<sizeof...(ContR) +1, 1>::type{};

	prepare_read(read_indices, cr);

	op();
}

} // namespace skepu

#endif // SKEPU_EXTERNAL_HPP
