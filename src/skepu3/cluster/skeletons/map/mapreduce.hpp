#pragma once
#ifndef SKEPU_CLUSTER_MAPREDUCE_HPP
#define SKEPU_CLUSTER_MAPREDUCE_HPP 1

#include <algorithm>
#include <omp.h>
#include <set>

#include <skepu3/cluster/common.hpp>
#include <skepu3/cluster/cluster.hpp>
#include "../skeleton_base.hpp"
#include "../skeleton_task.hpp"
#include "../skeleton_utils.hpp"

namespace skepu {
namespace backend {
namespace _starpu {

template<typename MapFunc, typename ReduceFunc>
struct map_reduce
{
	typedef ConditionalIndexForwarder<MapFunc::indexed, decltype(&MapFunc::CPU)>
		F;
	typedef typename ReduceFunc::Ret T;
	typedef index_dimension<
			typename std::conditional<MapFunc::indexed, typename MapFunc::IndexType,
			skepu::Index1D>::type>
		defaultDim;
	#pragma omp declare reduction(\
		mapred:T:omp_out=ReduceFunc::OMP(omp_out, omp_in))

	template<
		size_t... RI,
		size_t... EI,
		size_t... CI,
		typename Buffers,
		typename Iterator,
		typename ... CallArgs>
	auto static
	run(
		pack_indices<RI...>,
		pack_indices<EI...>,
		pack_indices<CI...>,
		Buffers && buffers,
		Iterator begin,
		size_t count,
		CallArgs && ... args) noexcept
	-> void
	{
		auto threads = starpu_combined_worker_get_size();
		omp_set_num_threads(threads);

		auto & res = std::get<0>(buffers)[0];
		res =
			F::forward(
				MapFunc::OMP,
				begin.index(),
				// Elwise elements are also raw pointers.
				std::get<EI>(buffers)[0]...,
				// Set MatRow to correct position. Does nothing for others.
				cluster::advance(std::get<CI>(buffers),begin.offset())...,
				args...);

		auto offset = begin.offset();
		#pragma omp parallel for reduction(mapred:res)
		for(size_t i = 1; i < count; ++i)
		{
			res =
				ReduceFunc::OMP(
					res,
					F::forward(
						MapFunc::OMP,
						(begin +i).index(),
						// Elwise elements are also raw pointers.
						std::get<EI>(buffers)[i]...,
						// Set MatRow to correct iition. Does nothing for others.
						cluster::advance(std::get<CI>(buffers),i)...,
						args...));
		}
	}

	template<
		size_t... RI,
		size_t... EI,
		size_t... CI,
		typename Buffers,
		typename ... CallArgs>
	auto static
	run(
		pack_indices<RI...>,
		pack_indices<EI...>,
		pack_indices<CI...>,
		Buffers && buffers,
		size_t offset,
		size_t count,
		size_t size_j,
		size_t size_k,
		size_t size_l,
		CallArgs && ... args) noexcept
	-> void
	{
		auto threads = starpu_combined_worker_get_size();
		omp_set_num_threads(threads);

		auto & res = std::get<0>(buffers)[0];
		res =
			F::forward(
				MapFunc::OMP,
				make_index(defaultDim{}, offset, size_j, size_k, size_l),
				// Set MatRow to correct position. Does nothing for others.
				cluster::advance(std::get<CI>(buffers), offset)...,
				args...);
		#pragma omp parallel for reduction(mapred:res)
		for(size_t i = 1; i < count; ++i)
		{
			res =
				ReduceFunc::OMP(
					res,
					F::forward(
						MapFunc::OMP,
						make_index(defaultDim{}, offset +i, size_j, size_k, size_l),
						// Set MatRow to correct position. Does nothing for others.
						cluster::advance(std::get<CI>(buffers), i)...,
						args...));
		}
	}
};

} // namespace _starpu

template<
	size_t arity,
	typename MapFunc,
	typename ReduceFunc,
	typename CUDAKernel,
	typename CUDAReduceKernel,
	typename CLKernel>
class MapReduce
: public SkeletonBase,
	private cluster::skeleton_task<
		_starpu::map_reduce<MapFunc, ReduceFunc>,
		std::tuple<typename MapFunc::Ret>,
		typename MapFunc::ElwiseArgs,
		typename MapFunc::ContainerArgs,
		typename MapFunc::UniformArgs>
{
	static constexpr auto skeletonType = SkeletonType::MapReduce;

	typedef typename ReduceFunc::Ret Ret;
	typedef typename MapFunc::Ret MapRes;
	typedef std::tuple<Ret> ResultArg;
	typedef typename MapFunc::ElwiseArgs ElwiseArgs;
	typedef typename MapFunc::ContainerArgs ContainerArgs;
	typedef typename MapFunc::UniformArgs UniformArgs;

	typedef cluster::skeleton_task<
			_starpu::map_reduce<MapFunc, ReduceFunc>,
			ResultArg,
			ElwiseArgs,
			ContainerArgs,
			UniformArgs>
		skeleton_task;

	static constexpr bool prefers_matrix = MapFunc::prefersMatrix;

	static constexpr size_t numArgs =
		MapFunc::totalArity - (MapFunc::indexed ? 1 : 0);
	static constexpr size_t anyArity =
		std::tuple_size<typename MapFunc::ContainerArgs>::value;

	static constexpr typename make_pack_indices<arity, 0>::type elwise_indices{};
	static constexpr typename make_pack_indices<arity + anyArity, arity>::type
		any_indices{};
	static constexpr typename make_pack_indices<numArgs, arity + anyArity>::type
		const_indices{};
	auto static constexpr ptag_indices =
		typename make_pack_indices<anyArity>::type{};

	typedef index_dimension<
			typename std::conditional<MapFunc::indexed, typename MapFunc::IndexType,
			skepu::Index1D>::type>
		defaultDim;
	typedef typename parameter_type<
			MapFunc::indexed ? 1 : 0,
			decltype(&MapFunc::CPU)>::type
		First;


	size_t default_size_i;
	size_t default_size_j;
	size_t default_size_k;
	size_t default_size_l;
	Ret m_start{};
	Ret m_result;
	std::vector<starpu_data_handle_t> m_result_handles;

public:
	MapReduce(CUDAKernel, CUDAReduceKernel)
	: skeleton_task("Skepu MapReduce"), m_result_handles(cluster::mpi_size())
	{
		auto rank = cluster::mpi_rank();
		for(size_t i(0); i < cluster::mpi_size(); ++i)
		{
			auto & handle = m_result_handles[i];
			int home_node = -1;
			Ret * ptr = 0;
			if(i == rank)
			{
				home_node = STARPU_MAIN_RAM;
				ptr = &m_result;
			}

			starpu_variable_data_register(
				&handle, home_node, (uintptr_t)ptr, sizeof(Ret));
			starpu_mpi_data_register(
				handle, cluster::mpi_tag(), i);
		}
	}

	~MapReduce() noexcept
	{
		for(auto & handle : m_result_handles)
			starpu_data_unregister_no_coherency(handle);
	}

	void setStartValue(Ret val)
	{
		m_start = val;
	}

	void setDefaultSize(size_t i, size_t j = 0, size_t k = 0, size_t l = 0)
	{
		default_size_i = i;
		default_size_j = j;
		default_size_k = k;
		default_size_l = l;
	}

	template<
		template<class> class Container,
		typename ... CallArgs,
		REQUIRES(is_skepu_container<Container<First>>::value)>
	auto
	operator()(Container<First> const & arg1, CallArgs && ... args)
	-> Ret
	{
		return
			backendDispatch(
				elwise_indices,
				any_indices,
				const_indices,
				arg1.begin(),
				arg1.end(),
				arg1,
				std::forward<CallArgs>(args)...);
	}

	template<
		template<class> class Container,
		typename ... CallArgs,
		REQUIRES(is_skepu_container<Container<First>>::value)>
	auto
	operator()(Container<First> & arg1, CallArgs && ... args) noexcept
	-> Ret
	{
		return
			backendDispatch(
				elwise_indices,
				any_indices,
				const_indices,
				arg1.begin(),
				arg1.end(),
				arg1,
				std::forward<CallArgs>(args)...);
	}

	template<
		typename Iterator,
		typename ... CallArgs,
		REQUIRES(is_skepu_iterator<Iterator, First>::value)>
	auto
	operator()(Iterator begin, Iterator end, CallArgs && ... args) noexcept
	-> Ret
	{
		return
			backendDispatch(
				elwise_indices,
				any_indices,
				const_indices,
				begin,
				end,
				begin.getParent(),
				std::forward<CallArgs>(args)...);
	}

	template<typename T = void, typename ... CallArgs>
	auto
	operator()(CallArgs && ... args) noexcept
	-> Ret
	{
		static_assert(sizeof...(CallArgs) == numArgs,
			"Number of arguments not matching Map function");

		size_t size = default_size_i;
		if(defaultDim::value >= 2)
			size *= default_size_j;
		if(defaultDim::value >= 3)
			size *= default_size_k;
		if(defaultDim::value >= 4)
			size *= default_size_l;

		return
			backendDispatch(
				any_indices,
				const_indices,
				size,
				std::forward<CallArgs>(args)...);
	}

private:
	template<
		size_t ... EI,
		size_t ... AI,
		size_t ... CI,
		typename Iterator,
		typename ... CallArgs>
	auto
	backendDispatch(
		pack_indices<EI...> ei,
		pack_indices<AI...> ai,
		pack_indices<CI...> ci,
		Iterator begin,
		Iterator end,
		CallArgs && ... args) noexcept
	-> Ret
	{
		if(disjunction((get<EI, CallArgs...>(args...).size() < (end - begin))...))
			SKEPU_ERROR("Non-matching container sizes");

		return
			STARPU(
				ei,
				ai,
				ci,
				ptag_indices,
				begin,
				end,
				std::forward<CallArgs>(args)...);
	}

	template<
		size_t ... AI,
		size_t ... CI,
		typename ... CallArgs>
	auto
	backendDispatch(
		pack_indices<AI...> ai,
		pack_indices<CI...> ci,
		size_t size,
		CallArgs && ... args) noexcept
	-> Ret
	{
		return
			STARPU(
				ai,
				ci,
				ptag_indices,
				size,
				std::forward<CallArgs>(args)...);
	}

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
		pack_indices<CI...> ci,
		pack_indices<PI...>,
		Iterator begin,
		Iterator end,
		CallArgs && ... args) noexcept
	-> Ret
	{
		auto static constexpr pt = typename MapFunc::ProxyTags{};
		auto static constexpr cbai = make_pack_indices<2>::type{};
		pack_expand(
			skeleton_task::handle_container_arg(
				cont::getParent(get<AI>(args...)),
				std::get<PI>(pt))...);
		pack_expand((cont::getParent(get<EI>(args...)).partition(),0)...);

		std::set<size_t> scheduled_ranks;
		while(begin != end)
		{
			auto pos = begin.offset();
			auto task_size = std::min({
				(size_t)(end - begin),
				cont::getParent(get<EI>(args...)).block_count_from(pos)...});
			auto rank = starpu_mpi_data_get_rank(begin.getParent().handle_for(pos));
			auto handles =
				std::make_tuple(
					m_result_handles[rank],
					cont::getParent(get<EI>(args...)).handle_for(pos)...,
					skeleton_task::container_handle(
						cont::getParent(get<AI>(args...)),
						std::get<PI>(pt),
						pos)...);
			auto call_back_args = std::make_tuple(begin, task_size);

			skeleton_task::schedule(
				ci,
				cbai,
				handles,
				call_back_args,
				std::forward<CallArgs>(args)...);

			scheduled_ranks.insert(rank);
			begin += task_size;
		}

		Ret res = m_start;
		for(auto rank : scheduled_ranks)
		{
			auto & handle = m_result_handles[rank];
			starpu_mpi_get_data_on_all_nodes_detached(MPI_COMM_WORLD, handle);
			starpu_data_acquire(handle, STARPU_R);
			res = ReduceFunc::CPU(res, *(Ret *)starpu_data_get_local_ptr(handle));
			starpu_data_release(handle);
			starpu_mpi_cache_flush(MPI_COMM_WORLD, handle);
		}

		return res;
	}

	template<
		size_t ... AI,
		size_t ... CI,
		size_t ... PI,
		typename ... CallArgs>
	auto
	STARPU(
		pack_indices<AI...>,
		pack_indices<CI...> ci,
		pack_indices<PI...>,
		size_t size,
		CallArgs && ... args) noexcept
	-> Ret
	{
		auto static constexpr pt = typename MapFunc::ProxyTags{};
		auto static constexpr cbai = make_pack_indices<5>::type{};
		pack_expand(
			skeleton_task::handle_container_arg(
				cont::getParent(get<AI>(args...)),
				std::get<PI>(pt))...);
		size_t task_size = size / cluster::mpi_size();
		if(size - (task_size * cluster::mpi_size()))
			++task_size;
		size_t pos = 0;
		size_t rank = 0;

		for(; pos < size; pos += task_size, ++rank)
		{
			auto handles =
				std::make_tuple(
					m_result_handles[rank],
					skeleton_task::container_handle(
						cont::getParent(get<AI>(args...)).handle_for(pos),
						std::get<PI>(pt),
						pos)...);
			size_t count = std::min(task_size, size - (rank * task_size));
			auto call_back_args =
				std::make_tuple(
					pos,
					count,
					default_size_j,
					default_size_k,
					default_size_l);

			skeleton_task::schedule(
				ci,
				cbai,
				handles,
				call_back_args,
				std::forward<CallArgs>(args)...);
		}

		Ret res = m_start;
		for(auto handle : m_result_handles)
		{
			starpu_mpi_get_data_on_all_nodes_detached(MPI_COMM_WORLD, handle);
			starpu_data_acquire(handle, STARPU_R);
			res = ReduceFunc::CPU(res, *(Ret *)starpu_data_get_local_ptr(handle));
			starpu_data_release(handle);
			starpu_mpi_cache_flush(MPI_COMM_WORLD, handle);
		}

		return res;
	}
};

} // namespace backend
} // namespace skepu

#endif// SKEPU_CLUSTER_MAPREDUCE_HPP
