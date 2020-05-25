#pragma once
#ifndef SKEPU_STARPU_SKELETON_REDUCE_HPP
#define SKEPU_STARPU_SKELETON_REDUCE_HPP 1

#include <omp.h>
#include <set>
#include <vector>

#include <starpu_mpi.h>

#include <skepu3/cluster/cluster.hpp>
#include <skepu3/cluster/common.hpp>
#include "../skeleton_base.hpp"
#include "../skeleton_task.hpp"
#include "reduce_mode.hpp"

namespace skepu {
namespace backend {
namespace _starpu {

template<typename ReduceFunc>
struct reduce1d
{
	typedef typename ReduceFunc::Ret T;

	template<
		size_t... RI,
		size_t... EI,
		size_t... CI,
		typename Buffers>
	auto static
	run(
		pack_indices<RI...>,
		pack_indices<EI...>,
		pack_indices<CI...>,
		Buffers && buffers,
		size_t const count) noexcept
	-> void
	{
		#pragma omp declare \
			reduction(reducer:T:omp_out=ReduceFunc::OMP(omp_out,omp_in))
		auto & res = std::get<0>(buffers)[0];
		auto container = std::get<1>(buffers);

		res = container[0];
		#pragma omp parallel for \
			reduction(reducer:res) \
			num_threads(starpu_combined_worker_get_size())
		for(size_t i = 1; i < count; ++i)
			res = ReduceFunc::OMP(res,container[i]);
	}

	template<
		typename Buffers,
		size_t... RI,
		size_t... EI,
		size_t... CI>
	auto static
	run(
		pack_indices<RI...>,
		pack_indices<EI...>,
		pack_indices<CI...>,
		Buffers && buffers,
		size_t const count,
		size_t const width) noexcept
	-> void
	{
		auto res = std::get<0>(buffers);
		auto matrix = std::get<1>(buffers);

		#pragma omp parallel for \
			num_threads(starpu_combined_worker_get_size())
		for(size_t i = 0; i < count; ++i)
		{
			auto row = matrix + (i*width);
			T row_res = row[0];
			for(size_t j(1); j < width; ++j)
				row_res = ReduceFunc::OMP(row_res, row[j]);
			res[i] = row_res;
		}
	}
};

} // namespace _starpu

template<typename ReduceFunc, typename CUDAKernel, typename CLKernel>
class Reduce1D
: public SkeletonBase,
	protected cluster::skeleton_task<
		_starpu::reduce1d<ReduceFunc>,
		std::tuple<typename ReduceFunc::Ret>,
		std::tuple<
			typename std::remove_reference<
				decltype(get<0>(typename ReduceFunc::UniformArgs{}))>::type>,
		std::tuple<>,
		std::tuple<>>
{
public:
	typedef typename ReduceFunc::Ret T;
	typedef cluster::skeleton_task<
			_starpu::reduce1d<ReduceFunc>,
			std::tuple<typename ReduceFunc::Ret>,
			std::tuple<
				typename std::remove_reference<
					decltype(get<0>(typename ReduceFunc::UniformArgs{}))>::type>,
			std::tuple<>,
			std::tuple<>>
		skeleton_task;

	auto static constexpr uniform_indices =
		make_pack_indices<0>::type{};

private:
	ReduceMode m_mode = ReduceMode::RowWise;

protected:
	T m_start{};
	T m_result;
	std::vector<starpu_data_handle_t> m_result_handles;

public:
	//static constexpr auto skeletonType = SkeletonType::Reduce1D;
	typedef std::tuple<T> ResultArg;
	typedef std::tuple<T> ElwiseArgs;
	typedef std::tuple<> ContainerArgs;

	static constexpr bool prefers_matrix = false;

	Reduce1D(CUDAKernel) noexcept
	: skeleton_task("Reduce1D_ElWise"),
		m_start(T()),
		m_result(0),
		m_result_handles(skepu::cluster::mpi_size(),  0)
	{
		auto rank = skepu::cluster::mpi_rank();
		for(size_t i(0); i < skepu::cluster::mpi_size(); ++i)
		{
			int home_node = -1;
			T * result_ptr(0);
			if(i == rank)
			{
				home_node = STARPU_MAIN_RAM;
				result_ptr = &m_result;
			}
			starpu_variable_data_register(
				&m_result_handles[i],
				home_node,
				(uintptr_t)result_ptr,
				sizeof(T));
			starpu_mpi_data_register(
				m_result_handles[i],
				skepu::cluster::mpi_tag(),
				i);
		}
	}

	~Reduce1D() noexcept
	{
		for(auto & handle : m_result_handles)
			starpu_data_unregister_no_coherency(handle);
	}

	auto
	setReduceMode(ReduceMode mode) noexcept
	-> void
	{
		this->m_mode = mode;
	}

	auto
	setStartValue(T val) noexcept
	-> void
	{
		this->m_start = val;
	}

	template<typename... Args>
	auto tune(Args&&...) noexcept -> void {}

	template<template<class> class Container,
		REQUIRES_VALUE(is_skepu_container<Container<T>>)>
	auto
	operator()(Container<T> & arg) noexcept
	-> T
	{
		return backendDispatch(arg.begin(), arg.end());
	}

	template<typename Iterator,
		REQUIRES_VALUE(is_skepu_iterator<Iterator,T>)>
	auto
	operator()(Iterator begin, Iterator end) noexcept
	-> T
	{
		return backendDispatch(begin, end);
	}

	template<
		template<typename>class Vector,
		template<typename>class Matrix,
		REQUIRES_VALUE(is_skepu_vector<Vector<T>>),
		REQUIRES_VALUE(is_skepu_matrix<Matrix<T>>)>
	auto
	operator()(Vector<T> & res, Matrix<T> & arg) noexcept
	-> Vector<T>
	{
		backendDispatch(res, arg);
		return res;
	}

private:
	template<typename Iterator>
	auto
	backendDispatch(Iterator begin, Iterator end) noexcept
	-> T
	{
		return STARPU(begin, end);
	}

	template<typename Vector, typename Matrix>
	auto
	backendDispatch(Vector & res, Matrix & arg)
	-> void
	{
		switch(m_mode)
		{
		case ReduceMode::ColWise:
		{
			auto argt = arg;
			argt.transpose(0);
			STARPU(res, argt);
			break;
		}
		default:
			STARPU(res, arg);
		}
	}

	template<typename Iterator>
	auto
	STARPU(Iterator begin, Iterator const end) noexcept
	-> T
	{
		auto static constexpr cbai = typename make_pack_indices<1>::type{};
		begin.getParent().partition();
		std::set<int> scheduled_ranks;

		while(begin != end)
		{
			auto pos = begin.offset();
			auto count = begin.getParent().block_count_from(pos);
			auto handle = begin.getParent().handle_for(pos);
			auto rank = starpu_mpi_data_get_rank(handle);
			scheduled_ranks.insert(rank);
			auto handles =
				std::make_tuple(
					m_result_handles[rank],
					handle);
			auto call_back_args = std::make_tuple(count);

			skeleton_task::schedule(
				uniform_indices,
				cbai,
				handles,
				call_back_args);

			begin += count;
		}

		auto result{m_start};
		for(auto & i : scheduled_ranks)
		{
			auto & handle = m_result_handles[i];
			starpu_mpi_get_data_on_all_nodes_detached(MPI_COMM_WORLD, handle);
			starpu_data_acquire(handle, STARPU_RW);
			auto ptr = (T *)starpu_variable_get_local_ptr(handle);
			result = ReduceFunc::CPU(result, *ptr);
			starpu_data_release(handle);
			starpu_mpi_cache_flush(MPI_COMM_WORLD, handle);
		}

		return result;
	}

	template<typename Vector, typename Matrix>
	auto
	STARPU(Vector & res, Matrix & data)
	-> Vector
	{
		auto static constexpr cbai = typename make_pack_indices<2>::type{};
		auto & rp = cont::getParent(res);
		auto & dp = cont::getParent(data);
		auto cols = data.size_j();

		rp.partition();
		dp.partition();

		for(size_t i(0); i < res.size();)
		{
			auto count = rp.block_count_from(i);
			auto handles = std::make_tuple(
				rp.handle_for(i),
				dp.handle_for(i * cols));
			auto call_back_args = std::make_tuple(count,cols);

			skeleton_task::schedule(
				uniform_indices,
				cbai,
				handles,
				call_back_args);

			i += count;
		}

		return res;
	}
};

namespace _starpu {

template<typename row_red_func, typename col_red_func>
struct reduce2d
{
	typedef typename row_red_func::Ret T;

	template<
		size_t... RI,
		size_t... EI,
		size_t... CI,
		typename Buffers>
	auto static
	run(
		pack_indices<RI...>,
		pack_indices<EI...>,
		pack_indices<CI...>,
		Buffers && buffers,
		size_t const rows,
		size_t const width) noexcept
	-> void
	{
		#pragma omp declare \
			reduction(row_red:T:omp_out=row_red_func::OMP(omp_out,omp_in))
		#pragma omp declare \
			reduction(col_red:T:omp_out=col_red_func::OMP(omp_out,omp_in))
		auto & res = std::get<0>(buffers)[0];
		auto matrix = std::get<1>(buffers);

		res = matrix[0];
		#pragma omp parallel for \
			reduction(row_red:res) \
			num_threads(starpu_combined_worker_get_size())
		for(size_t j = 1; j < width; ++j)
			res = row_red_func::OMP(res,matrix[j]);

		#pragma omp parallel for \
			reduction(col_red:res) \
			num_threads(starpu_combined_worker_get_size())
		for(size_t i = 1; i < rows; ++i)
		{
			auto row_it = matrix + (i * width);
			auto row_end = row_it + width;
			auto res = *(row_it++);
			for(; row_it != row_end; ++row_it)
				res = row_red_func::OMP(res, *row_it);
		}
	}
};

} // namespace _starpu

template<
	typename ReduceFuncRowWise,
	typename ReduceFuncColWise,
	typename CUDARowWise,
	typename CUDAColWise,
	typename CLKernel>
class Reduce2D
: public Reduce1D<ReduceFuncRowWise, CUDARowWise, CLKernel>,
	protected cluster::skeleton_task<
		_starpu::reduce2d<ReduceFuncRowWise, ReduceFuncColWise>,
		std::tuple<typename ReduceFuncRowWise::Ret>,
		std::tuple<
			typename std::remove_reference<
				decltype(get<0>(typename ReduceFuncRowWise::UniformArgs{}))>::type>,
		std::tuple<>,
		std::tuple<>>
{
	typedef typename cluster::skeleton_task<
		_starpu::reduce2d<ReduceFuncRowWise, ReduceFuncColWise>,
		std::tuple<typename ReduceFuncRowWise::Ret>,
		std::tuple<
			typename std::remove_reference<
				decltype(get<0>(typename ReduceFuncRowWise::UniformArgs{}))>::type>,
		std::tuple<>,
		std::tuple<>> skeleton_task;
	typedef Reduce1D<ReduceFuncRowWise, CUDARowWise, CLKernel> row_reduce;
	typedef typename ReduceFuncRowWise::Ret T;

	auto static constexpr uniform_indices =
		make_pack_indices<0>::type{};

public:
	// static constexpr auto skeletonType = SkeletonType::Reduce2D;
	static constexpr bool prefers_matrix = true;

	Reduce2D(CUDARowWise row, CUDAColWise)
	: row_reduce(row), skeleton_task("Reduce2D")
	{}

	template<
		template<typename>class Vector,
		REQUIRES_VALUE(is_skepu_vector<Vector<T>>)>
	auto
	operator()(Vector<T> & arg)
	-> T
	{
		return row_reduce::operator()(arg);
	}

	template<
		template<typename>class Matrix,
		REQUIRES_VALUE(is_skepu_matrix<Matrix<T>>)>
	auto
	operator()(Matrix<T> & arg)
	-> T
	{
		// If we need backend dispatch in the future.
		return STARPU(arg);
	}

private:

	template<typename Matrix>
	auto
	STARPU(Matrix & m)
	-> T
	{
		auto static constexpr cbai = typename make_pack_indices<2>::type{};
		auto cols = m.size_j();
		auto mp = cont::getParent(m);
		mp.partition();
		std::set<int> scheduled_ranks;

		for(size_t i = 0; i < m.size_j();)
		{
			auto count = mp.getParent().block_count_from(i) / cols;
			auto handle = mp.getParent().handle_for(i);
			auto rank = starpu_mpi_data_get_rank(handle);
			scheduled_ranks.insert(rank);
			auto handles =
				std::make_tuple(
					row_reduce::m_result_handles[rank],
					handle);
			auto call_back_args = std::make_tuple(count, cols);

			skeleton_task::schedule(
				uniform_indices,
				cbai,
				handles,
				call_back_args);

			i += count;
		}

		auto result{row_reduce::m_start};
		for(auto & i : scheduled_ranks)
		{
			auto & handle = row_reduce::m_result_handles[i];
			starpu_mpi_get_data_on_all_nodes_detached(MPI_COMM_WORLD, handle);
			starpu_data_acquire(handle, STARPU_RW);
			auto ptr = (T *)starpu_variable_get_local_ptr(handle);
			result = ReduceFuncColWise::OMP(result, *ptr);
			starpu_data_release(handle);
			starpu_mpi_cache_flush(MPI_COMM_WORLD, handle);
		}

		return result;
	}
};

} // namespace backend
} // namespace skepu

#endif // SKEPU_STARPU_SKELETON_REDUCE_HPP
