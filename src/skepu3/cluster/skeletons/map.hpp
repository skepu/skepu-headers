#pragma once
#ifndef SKEPU_CLUSTER_MAP_INTERFACE_H
#define SKEPU_CLUSTER_MAP_INTERFACE_H 1

#include <omp.h>
#include <tuple>
#include <starpu_mpi.h>

#include "../cluster.hpp"
#include "../common.hpp"
#include "skeleton_task.hpp"
#include "skeleton_base.hpp"

namespace skepu {
namespace backend {
namespace _starpu {

template<typename MapFunc>
struct map_omp
{
	typedef ConditionalIndexForwarder<MapFunc::indexed, decltype(&MapFunc::CPU)>
		F;

	template<typename Container>
	auto static
	advance(Container & c, size_t) noexcept
	-> Container &
	{
		return c;
	}

	template<typename T>
	auto static
	advance(skepu::MatRow<T> & mr, size_t rows) noexcept
	-> skepu::MatRow<T>
	{
		skepu::MatRow<T> advanced;
		advanced.data = mr.data + (rows * mr.cols);
		advanced.cols = mr.cols;
		return advanced;
	}

	template<typename T>
	auto static
	advance(skepu::MatRow<T> const & mr, size_t rows) noexcept
	-> skepu::MatRow<T>
	{
		skepu::MatRow<T> advanced;
		advanced.data = mr.data + (rows * mr.cols);
		advanced.cols = mr.cols;
		return advanced;
	}

	template<
		typename Buffers,
		typename Iterator,
		size_t... RI,
		size_t... EI,
		size_t... CI,
		typename... CallArgs>
	auto static
	run(
		pack_indices<RI...>,
		pack_indices<EI...>,
		pack_indices<CI...>,
		Iterator begin,
		size_t count,
		Buffers && buffers,
		CallArgs &&... args) noexcept
	-> void
	{
		// We use the pointer from the buffers here since its handle is already
		// locked by StarPU.
		auto & res = std::get<0>(buffers);

		#pragma omp parallel for num_threads(starpu_combined_worker_get_size())
		for(size_t i = 0; i < count; ++i)
		{
			res[i] =
				F::forward(
					MapFunc::OMP,
					(begin +i).index(),
					// Elwise elements are also raw pointers.
					std::get<EI>(buffers)[i]...,
					// Set MatRow to correct position. Does nothing for others.
					advance(std::get<CI>(buffers), i)...,
					args...);
		}
		return;
	}
};

} // namespace _starpu

template<size_t arity, typename MapFunc, typename CUDAKernel, typename CLKernel>
class Map
: public SkeletonBase,
	private cluster::skeleton_task<
		_starpu::map_omp<MapFunc>,
		std::tuple<typename MapFunc::Ret>,
		typename MapFunc::ElwiseArgs,
		typename MapFunc::ContainerArgs,
		typename MapFunc::UniformArgs>
{
	typedef typename MapFunc::Ret T;

	typedef cluster::skeleton_task<
			_starpu::map_omp<MapFunc>,
			std::tuple<T>,
			typename MapFunc::ElwiseArgs,
			typename MapFunc::ContainerArgs,
			typename MapFunc::UniformArgs>
		skeleton_task;

	static constexpr size_t numArgs =
		MapFunc::totalArity - (MapFunc::indexed ? 1 : 0);
	static constexpr size_t anyArity =
		std::tuple_size<typename MapFunc::ContainerArgs>::value;

	CUDAKernel m_cuda_kernel;

	size_t default_size_x;
	size_t default_size_y;

public:
	static constexpr auto skeletonType = SkeletonType::Map;
	using ResultArg = std::tuple<T>;
	using ElwiseArgs = typename MapFunc::ElwiseArgs;
	using ContainerArgs = typename MapFunc::ContainerArgs;
	using UniformArgs = typename MapFunc::UniformArgs;
	static constexpr bool prefers_matrix = MapFunc::prefersMatrix;

	static constexpr
	typename make_pack_indices<arity, 0>::type elwise_indices{};
	static constexpr
	typename make_pack_indices<arity + anyArity, arity>::type any_indices{};
	static constexpr
	typename make_pack_indices<anyArity, 0>::type ptag_indices{};
	static constexpr
	typename make_pack_indices<numArgs, arity + anyArity>::type const_indices{};

	Map(CUDAKernel kernel) : skeleton_task("SkePU Map"), m_cuda_kernel(kernel) {}

	auto
	setDefaultSize(size_t x, size_t y = 0) noexcept
	-> void
	{
		this->default_size_x = x;
		this->default_size_y = y;
	}

	template<typename... Args>
	auto tune(Args&&...) noexcept -> void {}

	template<template<class> class Container, typename... CallArgs,
		REQUIRES(is_skepu_container<Container<T>>::value)>
	Container<T> &operator()(Container<T> &res, CallArgs&&... args)
	{
		static_assert(sizeof...(CallArgs) == numArgs,
			"Number of arguments not matching Map function");

		backendDispatch(
			elwise_indices,
			any_indices,
			const_indices,
			ptag_indices,
			res.begin(),
			res.end(),
			std::forward<CallArgs>(args)...);
		return res;
	}

	template<typename Iterator, typename... CallArgs,
		REQUIRES(is_skepu_iterator<Iterator, T>::value)>
	Iterator operator()(Iterator begin, Iterator end, CallArgs&&... args)
	{
		static_assert(sizeof...(CallArgs) == numArgs,
			"Number of arguments not matching Map function");

		backendDispatch(
			elwise_indices,
			any_indices,
			const_indices,
			ptag_indices,
			begin,
			end,
			std::forward<CallArgs>(args)...);
		return begin;
	}

	// TODO: Remove if possible. Archaic and weird..
	template<template<typename>class Container, typename... CallArgs,
		REQUIRES(is_skepu_container<Container<T>>::value)>
	Container<T> operator()(CallArgs&&... args)
	{
		static_assert(sizeof...(CallArgs) == numArgs,
			"Number of arguments not matching Map function");

		Container<T> res(default_size_x);
		backendDispatch(
			elwise_indices,
			any_indices,
			const_indices,
			ptag_indices,
			res.size(),
			res.begin(),
			std::forward<CallArgs>(args)...);
		return res;
	}

private:
	template<
		size_t ... EI,
		size_t ... AI,
		size_t ... CI,
		size_t ... PI,
		typename Iterator,
		typename ... CallArgs>
	auto
	STARPU(
		pack_indices<EI...>,
		pack_indices<AI...>,
		pack_indices<CI...>,
		pack_indices<PI...>,
		Iterator begin,
		Iterator end,
		CallArgs && ... args) noexcept
	-> void
	{
		// The proxy elements require the data to be gathered on all nodes.
		// Except for MatRow, which will be partitioned.
		auto constexpr static pt = typename MapFunc::ProxyTags{};
		pack_expand((
			skeleton_task::handle_container_arg(
				cont::getParent(get<AI>(args...)),
				std::get<PI>(pt)),0)...);

		// The result will be partitioned so that every rank can do some work.
		begin.getParent().partition();

		// And the same goes for the elwise arguments.
		// TODO: They should be realinged so that element i in the elwise is on the
		// same rank as element i in the result. This will ensure that the number of
		// tasks generated is as low as possible.
		pack_expand((cont::getParent(get<EI>(args...)).partition(),0)...);

		while(begin != end)
		{
			auto pos = begin.offset();
			auto task_count = std::min({
				(size_t)(end - begin),
				begin.getParent().block_count_from(pos),
				cont::getParent(get<EI>(args...)).block_count_from(pos)...});
			auto handles =
				std::make_tuple(
					begin.getParent().handle_for(pos),
					cont::getParent(get<EI>(args...)).handle_for(pos)...,
					skeleton_task::container_handle(
						cont::getParent(get<AI>(args...)),
						std::get<PI>(pt),
						pos)...);

			this->schedule(
				const_indices,
				begin,
				task_count,
				handles,
				std::forward<CallArgs>(args)...);

			begin += task_count;
		}
	}

	template<
		size_t ... EI,
		size_t ... AI,
		size_t ... CI,
		size_t ... PI,
		typename Iterator,
		typename ... CallArgs>
	auto
	backendDispatch(
		pack_indices<EI...> ei,
		pack_indices<AI...> ai,
		pack_indices<CI...> ci,
		pack_indices<PI...> pi,
		Iterator begin,
		Iterator end,
		CallArgs && ... args) noexcept
	-> void
	{
		/* TODO:
		 * This should be more involved with possible backend selection. We only
		 * have OMP at this point, så we just redirect the call.
		 */
		STARPU(
			ei,
			ai,
			ci,
			pi,
			begin,
			end,
			std::forward<CallArgs>(args)...);
	}
};

} // namespace backend
} // namespace skepu

#endif // SKEPU_CLUSTER_MAP_INTERFACE_H
