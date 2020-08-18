#pragma once
#ifndef SKEPU_CLUSTER_MAP_HPP
#define SKEPU_CLUSTER_MAP_HPP 1

#include <omp.h>
#include <tuple>
#include <starpu_mpi.h>

#include "../../cluster.hpp"
#include "../../common.hpp"
#include "../skeleton_base.hpp"
#include "../skeleton_task.hpp"
#include "../skeleton_utils.hpp"

namespace skepu {
namespace backend {
namespace _starpu {

template<typename MapFunc>
struct map_omp
{
	typedef ConditionalIndexForwarder<MapFunc::indexed, decltype(&MapFunc::CPU)>
		F;

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
		Buffers && buffers,
		Iterator begin,
		size_t count,
		CallArgs &&... args) noexcept
	-> void
	{
		#pragma omp parallel for num_threads(starpu_combined_worker_get_size())
		for(size_t i = 0; i < count; ++i)
		{
			auto res =
				F::forward(
					MapFunc::OMP,
					(begin +i).index(),
					// Elwise elements are also raw pointers.
					std::get<EI>(buffers)[i]...,
					// Set MatRow to correct position. Does nothing for others.
					cluster::advance(std::get<CI>(buffers), i)...,
					args...);
			std::tie(std::get<RI>(buffers)[i]...) = res;
		}
	}
};

} // namespace _starpu

template<size_t arity, typename MapFunc, typename CUDAKernel, typename CLKernel>
class Map
: public SkeletonBase,
	private cluster::skeleton_task<
		_starpu::map_omp<MapFunc>,
		typename cluster::result_tuple<typename MapFunc::Ret>::type,
		typename MapFunc::ElwiseArgs,
		typename MapFunc::ContainerArgs,
		typename MapFunc::UniformArgs>
{
	typedef typename MapFunc::Ret T;

	typedef cluster::skeleton_task<
			_starpu::map_omp<MapFunc>,
			typename cluster::result_tuple<typename MapFunc::Ret>::type,
			typename MapFunc::ElwiseArgs,
			typename MapFunc::ContainerArgs,
			typename MapFunc::UniformArgs>
		skeleton_task;

	static constexpr size_t inArgs =
		MapFunc::totalArity - (MapFunc::indexed ? 1 : 0);
	static constexpr size_t outArgs =
		MapFunc::outArity;
	static constexpr size_t numArgs =
		inArgs + outArgs;
	static constexpr size_t anyArity =
		std::tuple_size<typename MapFunc::ContainerArgs>::value;

	CUDAKernel m_cuda_kernel;

	static constexpr typename make_pack_indices<outArgs>::type out_indices{};
	static constexpr
		typename make_pack_indices<arity + outArgs, outArgs>::type elwise_indices{};
	static constexpr
			typename
				make_pack_indices<arity + anyArity + outArgs, arity + outArgs>::type
		any_indices{};
	static constexpr
		typename make_pack_indices<anyArity, 0>::type ptag_indices{};
	static constexpr
			typename make_pack_indices<numArgs, arity + anyArity + outArgs>::type
		const_indices{};

public:
	Map(CUDAKernel kernel) : skeleton_task("SkePU Map"), m_cuda_kernel(kernel) {}

	template<typename... Args>
	auto tune(Args&&...) noexcept -> void {}

	template<typename... CallArgs>
	auto
	operator()(CallArgs&&... args) noexcept
	-> decltype(get<0>(args...))
	{
		static_assert(sizeof...(CallArgs) == numArgs,
			"Number of arguments not matching Map function");

		auto & res = get<0>(args...);

		backendDispatch(
			out_indices,
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
	auto
	operator()(Iterator begin, Iterator end, CallArgs&&... args) noexcept
	-> Iterator
	{
		static_assert(sizeof...(CallArgs) == numArgs -1,
			"Number of arguments not matching Map function");

		backendDispatch(
			out_indices,
			elwise_indices,
			any_indices,
			const_indices,
			ptag_indices,
			begin,
			end,
			begin,
			std::forward<CallArgs>(args)...);
		return begin;
	}

private:
	template<
		size_t ... OI,
		size_t ... EI,
		size_t ... AI,
		size_t ... CI,
		size_t ... PI,
		typename Iterator,
		typename ... CallArgs>
	auto
	STARPU(
		pack_indices<OI...>,
		pack_indices<EI...>,
		pack_indices<AI...>,
		pack_indices<CI...> ci,
		pack_indices<PI...>,
		Iterator begin,
		Iterator end,
		CallArgs && ... args) noexcept
	-> void
	{
		auto constexpr static pt = typename MapFunc::ProxyTags{};
		auto static constexpr cbai = make_pack_indices<2>::type{};

		// The proxy elements require the data to be gathered on all nodes.
		// Except for MatRow, which will be partitioned.
		pack_expand((
			skeleton_task::handle_container_arg(
				cont::getParent(get<AI>(args...)),
				std::get<PI>(pt)),0)...);

		// The result will be partitioned.
		pack_expand((cont::getParent(get<OI>(args...)).partition(),0)...);

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
					cont::getParent(get<OI>(args...)).handle_for(pos)...,
					cont::getParent(get<EI>(args...)).handle_for(pos)...,
					skeleton_task::container_handle(
						cont::getParent(get<AI>(args...)),
						std::get<PI>(pt),
						pos)...);
			auto call_back_args = std::make_tuple(begin, task_count);

			this->schedule(
				ci,
				cbai,
				handles,
				call_back_args,
				std::forward<CallArgs>(args)...);

			begin += task_count;
		}
	}

	template<
		size_t ... OI,
		size_t ... EI,
		size_t ... AI,
		size_t ... CI,
		size_t ... PI,
		typename Iterator,
		typename ... CallArgs>
	auto
	backendDispatch(
		pack_indices<OI...> oi,
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
		 * have OMP at this point, so we just redirect the call.
		 */
		STARPU(
			oi,
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

#endif // SKEPU_CLUSTER_MAP_HPP
