#pragma once
#ifndef SKEPU_CLUSTER_SKELETON_UTILS_HPP
#define SKEPU_CLUSTER_SKELETON_UTILS_HPP 1

#include <cstddef>
#include <starpu.h>

#include "../containers/proxies.hpp"

namespace skepu {
namespace cluster {

template<typename T>
struct result_tuple { typedef std::tuple<T> type; };

template<typename ... T>
struct result_tuple<std::tuple<T...>> { typedef std::tuple<T...> type; };

template<typename Container>
auto inline
advance(Container & c, size_t) noexcept
-> Container &
{
	return c;
}

template<typename T>
auto inline
advance(skepu::MatRow<T> & mr, size_t rows) noexcept
-> skepu::MatRow<T>
{
	skepu::MatRow<T> advanced;
	advanced.data = mr.data + (rows * mr.cols);
	advanced.cols = mr.cols;
	return advanced;
}

template<typename T>
auto inline
advance(skepu::MatRow<T> const & mr, size_t rows) noexcept
-> skepu::MatRow<T>
{
	skepu::MatRow<T> advanced;
	advanced.data = mr.data + (rows * mr.cols);
	advanced.cols = mr.cols;
	return advanced;
}

template<typename T>
auto inline
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
auto inline
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
auto inline
prepare_buffer(MatRow<T> const *, void * ptr)
-> MatRow<T>
{
	MatRow<T> proxy;
	proxy.data = (T *)STARPU_VECTOR_GET_PTR(ptr);
	proxy.cols = STARPU_MATRIX_GET_NX(ptr);

	return proxy;
}

template<typename T>
auto inline
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
auto inline
prepare_buffer(Ten4<T> const *, void * ptr)
-> Ten4<T>
{
	return Ten4<T>(
		(T *)STARPU_TENSOR_GET_PTR(ptr),
		STARPU_TENSOR_GET_NT(ptr),
		STARPU_TENSOR_GET_NZ(ptr),
		STARPU_TENSOR_GET_NY(ptr),
		STARPU_TENSOR_GET_NX(ptr));
}

template<typename T>
auto inline
prepare_buffer(Vec<T> const *, void * ptr)
-> Vec<T>
{
	Vec<T> proxy;
	proxy.data = (T *)STARPU_VECTOR_GET_PTR(ptr);
	proxy.size = STARPU_VECTOR_GET_NX(ptr);

	return proxy;
}

} // namespace cluster
} // namespace skepu

#endif // SKEPU_CLUSTER_SKELETON_UTILS_HPP
