#pragma once
#ifndef SKEPU_CLUSTER_COMMON_HPP
#define SKEPU_CLUSTER_COMMON_HPP 1

#define REQUIRES(...) typename std::enable_if<(__VA_ARGS__), bool>::type = 0
#define REQUIRES_VALUE(...) \
	typename std::enable_if<__VA_ARGS__::value, bool>::type = 0
#define REQUIRES_DEF(...) typename std::enable_if<(__VA_ARGS__), bool>::type

#define MAX_SIZE ((size_t)-1)

#define VARIANT_OPENCL(block)
#define VARIANT_CUDA(block)
#define VARIANT_OPENMP(block)
#define VARIANT_CPU(block) block

#include <vector>
#include <iostream>
#include <utility>
#include <cassert>
#include <algorithm>
#include <functional>

#include "../impl/backend.hpp"
#include "../impl/meta_helpers.hpp"
#include "flush_mode.hpp"

namespace skepu
{
	// ----------------------------------------------------------------
	// sizes and indices structures
	// ----------------------------------------------------------------

	struct Index1D
	{
		size_t i;
	};

	struct Index2D
	{
		size_t row, col;
	};

	struct Index3D
	{
		size_t i, j, k;
	};

	struct Index4D
	{
		size_t i, j, k, l;
	};

	struct ProxyTag
	{
		struct Default {};
		struct MatRow {};
	};

Index1D make_index(
	std::integral_constant<int, 1>,
	size_t index,
	size_t,
	size_t,
	size_t)
{
	return Index1D{index};
}

Index2D make_index(
	std::integral_constant<int, 2>,
	size_t index,
	size_t size_j,
	size_t,
	size_t)
{
	return Index2D{ index / size_j, index % size_j };
}

Index3D make_index(
	std::integral_constant<int, 3>,
	size_t index,
	size_t size_j,
	size_t size_k,
	size_t)
{
	size_t ci = index / (size_j * size_k);
	index = index % (size_j * size_k);
	size_t cj = index / (size_k);
	index = index % (size_k);
	return Index3D{ ci, cj, index };
}

Index4D make_index(
	std::integral_constant<int, 4>,
	size_t index,
	size_t size_j,
	size_t size_k,
	size_t size_l)
{
	size_t ci = index / (size_j * size_k * size_l);
	index = index % (size_j * size_k * size_l);
	size_t cj = index / (size_k * size_l);
	index = index % (size_k * size_l);
	size_t ck = index / (size_l);
	index = index % (size_l);
	return Index4D{ ci, cj, ck, index };
}

std::ostream & operator<<(std::ostream &o, Index1D idx)
{
	return o << "Index1D(" << idx.i << ")";
}

std::ostream & operator<<(std::ostream &o, Index2D idx)
{
	return o << "Index2D(" << idx.row << ", " << idx.col << ")";
}

std::ostream & operator<<(std::ostream &o, Index3D idx)
{
	return o << "Index3D(" << idx.i << ", "  << idx.j << ", " << idx.k << ")";
}

std::ostream & operator<<(std::ostream &o, Index4D idx)
{
	return o
		<< "Index4D("
		<< idx.i
		<< ", "
		<< idx.j
		<< ", "
		<< idx.k
		<< ", "
		<< idx.l
		<< ")";
}

	// Container Regions (perhaps relocate)

	template<typename T>
	struct Region1D
	{
		int oi;
		size_t stride;
		const T *data;

		T operator()(int i)
		{
			return data[i * this->stride];
		}

		Region1D(int arg_oi, size_t arg_stride, const T *arg_data)
		: oi(arg_oi), stride(arg_stride), data(arg_data) {}
	};

	template<typename T>
	struct Region2D
	{
		int oi, oj;
		size_t stride;
		const T *data;

		T operator()(int i, int j)
		{
			return data[i * this->stride + j];
		}

		Region2D(int arg_oi, int arg_oj, size_t arg_stride, const T *arg_data)
		: oi(arg_oi), oj(arg_oj), stride(arg_stride), data(arg_data) {}
	};

	template<typename T>
	struct Region3D
	{
		int oi, oj, ok;
		size_t stride1, stride2;
		const T *data;

		T operator()(int i, int j, int k)
		{
			return data[i * this->stride1 * this->stride2 + j * this->stride2 + k];
		}

		Region3D(int arg_oi, int arg_oj, int arg_ok, size_t arg_stride1, size_t arg_stride2, const T *arg_data)
		: oi(arg_oi), oj(arg_oj), ok(arg_ok), stride1(arg_stride1), stride2(arg_stride2), data(arg_data) {}
	};

	template<typename T>
	struct Region4D
	{
		int oi, oj, ok, ol;
		size_t stride1, stride2, stride3;
		const T *data;

		T operator()(int i, int j, int k, int l)
		{
			return data[i * this->stride1 * this->stride2 * this->stride3 + j * this->stride2 * this->stride3 + k * this->stride3 + l];
		}

		Region4D(int arg_oi, int arg_oj, int arg_ok, int arg_ol, size_t arg_stride1, size_t arg_stride2, size_t arg_stride3, const T *arg_data)
		: oi(arg_oi), oj(arg_oj), ok(arg_ok), ol(arg_ol), stride1(arg_stride1), stride2(arg_stride2), stride3(arg_stride3), data(arg_data) {}
	};


	template<typename T>
	struct region_type {};

	template<typename T>
	struct region_type<Region1D<T>> { using type = T; };

	template<typename T>
	struct region_type<Region2D<T>> { using type = T; };

	template<typename T>
	struct region_type<Region3D<T>> { using type = T; };

	template<typename T>
	struct region_type<Region4D<T>> { using type = T; };


	enum class AccessMode
	{
		Read,
		Write,
		ReadWrite,
		None
	};

	static inline constexpr bool hasReadAccess(AccessMode m)
	{
		return m == AccessMode::Read || m == AccessMode::ReadWrite;
	}

	static inline constexpr bool hasWriteAccess(AccessMode m)
	{
		return m == AccessMode::Write || m == AccessMode::ReadWrite;
	}

	enum class SkeletonType
	{
		Map,
		MapReduce,
		MapPairs,
		MapPairsReduce,
		Reduce1D,
		Reduce2D,
		Scan,
		MapOverlap1D,
		MapOverlap2D,
		MapOverlap3D,
		MapOverlap4D,
		Call,
	};

/* To be able to use getParent on containers. Those are private in MPI. */
struct cont
{
	template<typename Container>
	static auto
	getParent(Container && c)
	-> decltype(c.getParent())
	{
		return c.getParent();
	}
};

	// For multiple return Map variants
	template<typename... args>
	using multiple = std::tuple<args...>;

	template <typename... Args>
	auto ret(Args&&... args)
	-> decltype(std::make_tuple(std::forward<Args>(args)...))
	{
		return std::make_tuple(std::forward<Args>(args)...);
	}

	inline size_t elwise_i(std::tuple<>) { return 0; }
	inline size_t elwise_j(std::tuple<>) { return 0; }
	inline size_t elwise_k(std::tuple<>) { return 0; }
	inline size_t elwise_l(std::tuple<>) { return 0; }

	template<typename... Args>
	inline size_t elwise_i(std::tuple<Args...> &t)
 	{
		return cont::getParent(std::get<0>(t)).size_i();
 	}

 	template<typename... Args>
	inline size_t elwise_j(std::tuple<Args...> &t)
 	{
		return cont::getParent(std::get<0>(t)).size_j();
 	}

	template<typename... Args>
	inline size_t elwise_k(std::tuple<Args...> &t)
	{
		return cont::getParent(std::get<0>(t)).size_k();
	}

	template<typename... Args>
	inline size_t elwise_l(std::tuple<Args...> &t)
	{
		return cont::getParent(std::get<0>(t)).size_l();
	}

	// ----------------------------------------------------------------
	// is_skepu_{vector|matrix|container} trait classes
	// ----------------------------------------------------------------

	template<typename T>
	struct is_skepu_vector: std::false_type {};

	template<typename T>
	struct is_skepu_matrix: std::false_type {};

	template<typename T>
	struct is_skepu_tensor3: std::false_type {};

	template<typename T>
	struct is_skepu_tensor4: std::false_type {};

	template<typename T>
	struct is_skepu_container:
		std::integral_constant<bool,
			is_skepu_vector<typename std::remove_cv<typename std::remove_reference<T>::type>::type>::value ||
			is_skepu_matrix<typename std::remove_cv<typename std::remove_reference<T>::type>::type>::value ||
			is_skepu_tensor3<typename std::remove_cv<typename std::remove_reference<T>::type>::type>::value ||
			is_skepu_tensor4<typename std::remove_cv<typename std::remove_reference<T>::type>::type>::value> {};

	/** Check that all types in a parameter pack are SkePU containers. */
	template<typename ...> struct are_skepu_containers;

	/* Empty pack is true. */
	template<> struct are_skepu_containers<> : public std::true_type {};

	/* Check that first is a SkePU container and recurse the rest. */
	template<typename CAR, typename ... CDR>
	struct are_skepu_containers<CAR, CDR...>
	: std::integral_constant<bool,
			is_skepu_container<CAR>::value
			&& are_skepu_containers<CDR...>::value>
	{};

	template<typename T>
	struct is_skepu_vector_proxy: std::false_type {};

	template<typename T>
	struct is_skepu_matrix_proxy: std::false_type {};

	template<typename T>
	struct is_skepu_tensor3_proxy: std::false_type {};

	template<typename T>
	struct is_skepu_tensor4_proxy: std::false_type {};

	template<typename T>
	struct is_skepu_container_proxy:
		std::integral_constant<bool,
			is_skepu_vector_proxy<typename std::remove_cv<typename std::remove_reference<T>::type>::type>::value ||
			is_skepu_matrix_proxy<typename std::remove_cv<typename std::remove_reference<T>::type>::type>::value ||
			is_skepu_tensor3_proxy<typename std::remove_cv<typename std::remove_reference<T>::type>::type>::value ||
			is_skepu_tensor4_proxy<typename std::remove_cv<typename std::remove_reference<T>::type>::type>::value>  {};

	// ----------------------------------------------------------------
	// is_skepu_iterator trait class
	// ----------------------------------------------------------------

	template<typename T, typename Ret>
	struct is_skepu_iterator: std::false_type {};

 	// ----------------------------------------------------------------
	// index trait classes for skepu::IndexND (N in [1,2,3,4])
 	// ----------------------------------------------------------------

	// returns the dimensionality of an index type.
	// that is, if the type is skepu::IndexND, then returns N, else 0.
 	template<typename T>
	struct index_dimension: std::integral_constant<int, 0>{};

 	template<>
	struct index_dimension<Index1D>: std::integral_constant<int, 1>{};

 	template<>
	struct index_dimension<Index2D>: std::integral_constant<int, 2>{};

 	template<>
	struct index_dimension<Index3D>: std::integral_constant<int, 3>{};

 	template<>
	struct index_dimension<Index4D>: std::integral_constant<int, 4>{};

	// true iff T is a SkePU index type
	template<typename T>
	struct is_skepu_index: bool_constant<index_dimension<T>::value != 0>{};

	// true iff first element of Args is SkePU index type
 	template<typename... Args>
	struct is_indexed
	: bool_constant<is_skepu_index<
			typename first_element<Args...>::type>::value>{};

 	template<>
	struct is_indexed<>: std::false_type{};

	// ----------------------------------------------------------------
	// matrix row proxy trait class
	// ----------------------------------------------------------------

	template<typename T>
	struct proxy_tag {
		using type = ProxyTag::Default;
	};

	// ----------------------------------------------------------------
	// smart container size extractor
	// ----------------------------------------------------------------

	inline std::tuple<size_t>
	size_info(index_dimension<skepu::Index1D>, size_t i, size_t, size_t, size_t)
	{
		return {i};
	}

	inline std::tuple<size_t, size_t>
	size_info(index_dimension<skepu::Index2D>, size_t i, size_t j, size_t, size_t)
	{
		return {i, j};
	}

	inline std::tuple<size_t, size_t, size_t>
	size_info(
		index_dimension<skepu::Index3D>, size_t i, size_t j, size_t k, size_t)
	{
		return {i, j, k};
	}

	inline std::tuple<size_t, size_t, size_t, size_t>
	size_info(
		index_dimension<skepu::Index4D>, size_t i, size_t j, size_t k, size_t l)
	{
		return {i, j, k, l};
	}

	template<typename Index, typename... Args>
	inline auto
	size_info(Index, size_t, size_t, size_t, size_t, Args&&... args)
	-> decltype(get<0, Args...>(args...).getParent().size_info())
	{
		return get<0, Args...>(args...).getParent().size_info();
	}

	// ----------------------------------------------------------------
	// Smart Container Coherency Helpers
	// ----------------------------------------------------------------

	/*
	 * Base case for recursive variadic flush.
	 */
	template<FlushMode mode>
	void flush() {}

	/*
	 *
	 */
	template<FlushMode mode = FlushMode::Default, typename First, typename... Args>
	void flush(First&& first, Args&&... args)
	{
		first.flush(mode);
		flush<mode>(std::forward<Args>(args)...);
	}


	// ----------------------------------------------------------------
	// ConditionalIndexForwarder utility structure
	// ----------------------------------------------------------------

 	template<bool indexed, typename Func>
 	struct ConditionalIndexForwarder
 	{
 		using Ret = typename return_type<Func>::type;

		// Forward index

		template<typename Index, typename... CallArgs, REQUIRES(is_skepu_index<Index>::value && indexed)>
		static Ret forward(Func func, Index i, CallArgs&&... args)
 		{
 			return func(i, std::forward<CallArgs>(args)...);
 		}

		template<typename Index, typename... CallArgs, REQUIRES(is_skepu_index<Index>::value && indexed)>
		static Ret forward_device(Func func, Index i, CallArgs&&... args)
 		{
 			return func(i, std::forward<CallArgs>(args)...);
 		}

		// Do not forward index

		template<typename Index, typename... CallArgs, REQUIRES(is_skepu_index<Index>::value && !indexed)>
		static Ret forward(Func func, Index, CallArgs&&... args)
 		{
 			return func(std::forward<CallArgs>(args)...);
 		}

		template<typename Index, typename... CallArgs, REQUIRES(is_skepu_index<Index>::value && !indexed)>
		static Ret forward_device(Func func, Index, CallArgs&&... args)
 		{
 			return func(std::forward<CallArgs>(args)...);
 		}
 	};

} // namespace skepu

#endif // SKEPU_CLUSTER_COMMON_HPP
