#pragma once
#ifndef SKEPU_CLUSTER_SKELETONS_SKELETON_TASK_HPP
#define SKEPU_CLUSTER_SKELETONS_SKELETON_TASK_HPP 1

#include <starpu.h>
#include <tuple>

#include "../common.hpp"
#include "../containers/proxies.hpp"
#include "../helpers.hpp"
#include "../handle_modes.hpp"
#include "../task_schedule_wrapper.hpp"

namespace skepu {
namespace cluster {

/** Encapsulating class for starpu tasks.
 * This class encapsulates some data access patterns common to all skeletons,
 * using the curiously recccuring template pattern. This places some
 * requirements on its subclasses.
 *
 * Each parameter is a tuple of the underlying datatype of each
 * desired buffer.
 *
 * \author Henrik Henriksson
 * \author Johan Ahlqvist
 *
 * \tparam ResultArgs			STARPU_W
 * \tparam ElwiseArgs			STARPU_R
 * \tparam ContainerArgs	STARPU_R
 * \tparam UniformArgs
 *
 */
template<
	typename CallBackFn,
	typename ResultArgs,
	typename ElwiseArgs,
	typename ContainerArgs,
	typename UniformArgs>
class skeleton_task
{
	starpu_codelet cl;
	starpu_perfmodel perf_model;

	using HandleT =
		decltype(std::tuple_cat(ResultArgs{}, ElwiseArgs{}, ContainerArgs{}));

	static constexpr size_t n_result = std::tuple_size<ResultArgs>::value;
	static constexpr size_t n_elwise = std::tuple_size<ElwiseArgs>::value;
	static constexpr size_t n_container = std::tuple_size<ContainerArgs>::value;
	static constexpr size_t n_uniform = std::tuple_size<UniformArgs>::value;
	static constexpr size_t n_handles = n_result + n_elwise + n_container;

	static constexpr
	typename make_pack_indices<n_handles, 0>::type handle_indices{};

	static constexpr
	typename make_pack_indices<n_result, 0>::type result_handle_indices{};

	static constexpr
	typename make_pack_indices<n_result+n_elwise, n_result>::type
	elwise_handle_indices{};

	static constexpr
	typename make_pack_indices<
		n_result+n_elwise+n_container,n_result+n_elwise>::type
	container_handle_indices{};

	static constexpr typename make_pack_indices<n_result, 0>::type ri{};
	static constexpr typename make_pack_indices<n_elwise, 0>::type ei{};
	static constexpr typename make_pack_indices<n_container, 0>::type ci{};
	static constexpr typename make_pack_indices<n_uniform, 0>::type ui{};

	static constexpr size_t n_arg = n_result+n_elwise+n_container+n_uniform;

protected:
	skeleton_task(char const * name)
	{
		// Initialize performance model
		memset(&perf_model, 0, sizeof(starpu_perfmodel));
		starpu_perfmodel_init(&perf_model);
		perf_model.type = STARPU_HISTORY_BASED;

		// Not really used for now.
		perf_model.symbol = name;

		starpu_codelet_init(&cl);
		cl.nbuffers = n_handles;
		cl.max_parallelism = INT_MAX;
		cl.type = STARPU_FORKJOIN; // For OpenMP
		cl.where = STARPU_CPU;
		//cl.cpu_funcs[0] = starpu_task_callback;
		cl.cpu_funcs_name[0] = name;
		cl.modes[0] = STARPU_RW;

		// Performance model not needed?
		// Creates a problem with starpu_shutdown if used with MPI.
		// cl.model = &perf_model;

		helpers::set_codelet_read_only_modes(handle_indices, cl);
		for(size_t i {}; i < n_result; ++i)
			cl.modes[i] = STARPU_RW;
	}

	template<typename Container, typename ProxyTag>
	auto
	container_handle(Container & c, ProxyTag, size_t)
	-> starpu_data_handle_t
	{
		return c.local_storage_handle();
	}

	template<typename Container>
	auto
	container_handle(Container & c, skepu::ProxyTag::MatRow const &, size_t row)
	-> starpu_data_handle_t
	{
		return c.handle_for_row(row);
	}

	template<typename Container, typename ProxyTag>
	auto
	handle_container_arg(Container & c, ProxyTag) noexcept
	-> void
	{
		c.allgather();
	}

	template<typename Container>
	auto
	handle_container_arg(Container & m, skepu::ProxyTag::MatRow)
	-> void
	{
		m.partition();
	}

	template<
		size_t ... UI,
		typename Iterator,
		typename ... Handles,
		typename ... Args>
	auto
	schedule(
		pack_indices<UI...>,
		Iterator begin,
		size_t count,
		std::tuple<Handles...> & handles,
		Args && ... args)
	-> void
	{
		cl.cpu_funcs[0] = starpu_task_callback<Iterator>;
		auto modes = helpers::modes_from_codelet(cl);
		auto uniform = std::make_tuple(begin, count, get<UI>(args...)...);

		helpers::schedule_task(&cl, modes, handles, uniform);
	}

private:
	template<typename Iterator>
	auto static
	starpu_task_callback(void ** buffers, void * args) noexcept
	-> void
	{
		handle_callback<Iterator>(
			handle_indices,
			ri,
			ei,
			ci,
			ui,
			buffers,
			args);
	}

	template<
		typename Iterator,
		size_t ... HI,
		size_t ... RI,
		size_t ... EI,
		size_t ... CI,
		size_t ... UI>
	auto static
	handle_callback(
		pack_indices<HI...>,
		pack_indices<RI...>,
		pack_indices<EI...>,
		pack_indices<CI...>,
		pack_indices<UI...>,
		void ** buffers,
		void * args) noexcept
	-> void
	{
		Iterator begin;
		size_t count;
		UniformArgs uniform_args;
		helpers::extract_constants(
			args,
			begin,
			count,
			std::get<UI>(uniform_args)...);

		using buffers_type = decltype(std::tuple_cat(
		//typedef decltype(std::tuple_cat(
				std::make_tuple(
					typename std::add_pointer<
						decltype(std::get<RI>(ResultArgs{}))>::type{}...),
				std::make_tuple(
					typename std::add_pointer<
						decltype(std::get<EI>(ElwiseArgs{}))>::type{}...),
				ContainerArgs{}))
		;//	buffers_type;

		buffers_type containers(
			prepare_buffer(
				(typename std::tuple_element<HI, buffers_type>::type *)0,
				buffers[HI])...);
		CallBackFn::run(
			result_handle_indices,
			elwise_handle_indices,
			container_handle_indices,
			begin,
			count,
			std::forward<buffers_type>(containers),
			std::get<UI>(uniform_args)...);
	}

	template<typename T>
	auto static
	prepare_buffer(T const *, void * ptr)
	-> T
	{
		auto type_id = *((starpu_data_interface_id *)ptr);
		switch(type_id)
		{
			case STARPU_MATRIX_INTERFACE_ID:
				return (T)STARPU_MATRIX_GET_PTR(ptr);
			case STARPU_BLOCK_INTERFACE_ID:
				return (T)STARPU_BLOCK_GET_PTR(ptr);
			case STARPU_VECTOR_INTERFACE_ID:
				return (T)STARPU_VECTOR_GET_PTR(ptr);
			case STARPU_VARIABLE_INTERFACE_ID:
				return (T)STARPU_VARIABLE_GET_PTR(ptr);
			default:
				std::cerr << "[SkePU][skeleton_task] "
					"Unable to determine StarPU buffer type in task.\n";
				std::abort();
		};
	}

	template<typename T>
	auto static
	prepare_buffer(Mat<T> const *, void * ptr)
	-> Mat<T>
	{
		Mat<T> proxy;
		proxy.data = (T *)STARPU_VECTOR_GET_PTR(ptr);
		proxy.rows = STARPU_MATRIX_GET_NY(ptr);
		proxy.cols = STARPU_MATRIX_GET_NX(ptr);

		return proxy;
	}

	template<typename T>
	auto static
	prepare_buffer(MatRow<T> const *, void * ptr)
	-> MatRow<T>
	{
		MatRow<T> proxy;
		proxy.data = (T *)STARPU_VECTOR_GET_PTR(ptr);
		proxy.cols = STARPU_MATRIX_GET_NX(ptr);

		return proxy;
	}

	template<typename T>
	auto static
	prepare_buffer(Ten3<T> const *, void * ptr)
	-> Ten3<T>
	{
		return Ten3<T>(
			(T *)STARPU_BLOCK_GET_PTR(ptr),
			STARPU_BLOCK_GET_NZ(ptr),
			STARPU_BLOCK_GET_NY(ptr),
			STARPU_BLOCK_GET_NX(ptr));
	}

	template<typename T>
	auto static
	prepare_buffer(Vec<T> const *, void * ptr)
	-> Vec<T>
	{
		Vec<T> proxy;
		proxy.data = (T *)STARPU_VECTOR_GET_PTR(ptr);
		proxy.size = STARPU_VECTOR_GET_NX(ptr);

		return proxy;
	}
};

} // namespace cluster
} // namespace skepu

#endif // SKEPU_CLUSTER_SKELETONS_SKELETON_TASK_HPP