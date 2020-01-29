#pragma once

#define REQUIRES(...) typename std::enable_if<(__VA_ARGS__), bool>::type = 0
#define REQUIRES_VALUE(...) typename std::enable_if<__VA_ARGS__::value, bool>::type = 0
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
		Reduce1D,
		Reduce2D,
		Scan,
		MapOverlap1D,
		MapOverlap2D,
		MapOverlap3D,
		MapOverlap4D,
		Call,
	};
	
	
	// For multiple return Map variants
	
	template<typename... args>
	using multiple = std::tuple<args...>;
	
	template <typename... Args>
	auto ret(Args&&... args) -> decltype(std::make_tuple(std::forward<Args>(args)...)) {
		return std::make_tuple(std::forward<Args>(args)...);
	}
	
	
#ifdef SKEPU_OPENCL
	
	/*!
	 * helper to return data type in a string format using template specialication technique.
	 * Compile-time error if no overload is found.
	 */
	template<typename T>
	inline std::string getDataTypeCL();
	
	template<> inline std::string getDataTypeCL<char>          () { return "char";           }
	template<> inline std::string getDataTypeCL<unsigned char> () { return "unsigned char";  }
	template<> inline std::string getDataTypeCL<short>         () { return "short";          }
	template<> inline std::string getDataTypeCL<unsigned short>() { return "unsigned short"; }
	template<> inline std::string getDataTypeCL<int>           () { return "int";            }
	template<> inline std::string getDataTypeCL<unsigned int>  () { return "unsigned int";   }
	template<> inline std::string getDataTypeCL<long>          () { return "long";           }
	template<> inline std::string getDataTypeCL<unsigned long> () { return "unsigned long";  }
	template<> inline std::string getDataTypeCL<float>         () { return "float";          }
	template<> inline std::string getDataTypeCL<double>        () { return "double";         }
	
#endif
	
	// Dummy base class for sequential skeleton classes.
	// Includes empty member functions which has no meaning in a sequential context.
	class SeqSkeletonBase
	{
	public:
		void setBackend(BackendSpec) {}
		void resetBackend() {}
		void setExecPlan(ExecPlan *plan)
		{
			delete plan;
		}
		
		template<typename... Args>
		void tune(Args&&... args) { }
	};
}

#include "meta_helpers.hpp"
#include "skepu3/vector.hpp"
#include "skepu3/matrix.hpp"
#include "skepu3/tensor.hpp"
#include "skepu3/sparse_matrix.hpp"

namespace skepu
{
	inline size_t elwise_width(std::tuple<>)
	{
		return 0;
	}
	
	template<typename... Args>
	inline size_t elwise_width(std::tuple<Args...> &t)
	{
		return std::get<0>(t).getParent().total_cols();
	}
	
	// ----------------------------------------------------------------
	// is_skepu_{vector|matrix|container} trait classes
	// ----------------------------------------------------------------
	
	template<typename T>
	struct is_skepu_vector: std::false_type {};
	
	template<typename T>
	struct is_skepu_vector<skepu::Vector<T>>: std::true_type {};
	
	
	template<typename T>
	struct is_skepu_matrix: std::false_type {};
	
	template<typename T>
	struct is_skepu_matrix<skepu::Matrix<T>>: std::true_type {};
	
	template<typename T>
	struct is_skepu_matrix<skepu::SparseMatrix<T>>: std::true_type {};
	
	
	template<typename T>
	struct is_skepu_tensor3: std::false_type {};
	
	template<typename T>
	struct is_skepu_tensor3<skepu::Tensor3<T>>: std::true_type {};
	
	
	template<typename T>
	struct is_skepu_tensor4: std::false_type {};
	
	template<typename T>
	struct is_skepu_tensor4<skepu::Tensor4<T>>: std::true_type {};
	
	
	template<typename T>
	struct is_skepu_container:
		std::integral_constant<bool,
			is_skepu_vector<typename std::remove_cv<typename std::remove_reference<T>::type>::type>::value ||
			is_skepu_matrix<typename std::remove_cv<typename std::remove_reference<T>::type>::type>::value ||
			is_skepu_tensor3<typename std::remove_cv<typename std::remove_reference<T>::type>::type>::value ||
			is_skepu_tensor4<typename std::remove_cv<typename std::remove_reference<T>::type>::type>::value> {};
		
		
		
	template<typename T>
	struct is_skepu_vector_proxy: std::false_type {};
	
	template<typename T>
	struct is_skepu_vector_proxy<skepu::Vec<T>>: std::true_type {};
	
	template<typename T>
	struct is_skepu_matrix_proxy: std::false_type {};
	
	template<typename T>
	struct is_skepu_matrix_proxy<skepu::Mat<T>>: std::true_type {};
	
	template<typename T>
	struct is_skepu_matrix_proxy<skepu::MatRow<T>>: std::true_type {};
	
	template<typename T>
	struct is_skepu_matrix_proxy<skepu::SparseMat<T>>: std::true_type {};
	
	template<typename T>
	struct is_skepu_tensor3_proxy: std::false_type {};
	
	template<typename T>
	struct is_skepu_tensor3_proxy<skepu::Ten3<T>>: std::true_type {};
	
	template<typename T>
	struct is_skepu_tensor4_proxy: std::false_type {};
	
	template<typename T>
	struct is_skepu_tensor4_proxy<skepu::Ten4<T>>: std::true_type {};
	
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
	struct is_skepu_iterator: bool_constant<
		std::is_same<T, typename Vector<Ret>::iterator>::value ||
		std::is_same<T, typename Vector<Ret>::const_iterator>::value ||
		std::is_same<T, typename Matrix<Ret>::iterator>::value
	> {};
	
	
	// ----------------------------------------------------------------
	// is_skepu_index trait class
	// ----------------------------------------------------------------
	
	template<typename T>
	struct is_skepu_index: std::false_type{};
	
	template<>
	struct is_skepu_index<Index1D>: std::true_type{};
	
	template<>
	struct is_skepu_index<Index2D>: std::true_type{};
	
	template<>
	struct is_skepu_index<Index3D>: std::true_type{};
	
	template<>
	struct is_skepu_index<Index4D>: std::true_type{};
	
	
	template<typename... Args>
	struct is_indexed
	: bool_constant<is_skepu_index<typename pack_element<0, Args...>::type>::value> {};
	
	template<>
	struct is_indexed<>
	: std::false_type{};
	
	// ----------------------------------------------------------------
	// matrix row proxy trait class
	// ----------------------------------------------------------------
	
	template<typename T>
	struct proxy_tag {
		using type = ProxyTag::Default;
	};
	
	template<typename T>
	struct proxy_tag<MatRow<T>> {
		using type = ProxyTag::MatRow;
	};
	
	
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
		
		template<typename... CallArgs>
		static Ret forward(Func func, Index1D i, CallArgs&&... args)
		{
			return func(i, std::forward<CallArgs>(args)...);
		}
		
		template<typename... CallArgs>
		static Ret forward(Func func, Index2D i, CallArgs&&... args)
		{
			return func(i, std::forward<CallArgs>(args)...);
		}
		
		template<typename... CallArgs>
		static Ret forward(Func func, Index3D i, CallArgs&&... args)
		{
			return func(i, std::forward<CallArgs>(args)...);
		}
		
		template<typename... CallArgs>
		static Ret forward(Func func, Index4D i, CallArgs&&... args)
		{
			return func(i, std::forward<CallArgs>(args)...);
		}
		
		
		template<typename... CallArgs>
		static Ret forward_device(Func func, Index1D i, CallArgs&&... args)
		{
			return func(i, std::forward<CallArgs>(args)...);
		}
		
		template<typename... CallArgs>
		static Ret forward_device(Func func, Index2D i, CallArgs&&... args)
		{
			return func(i, std::forward<CallArgs>(args)...);
		}
		
		template<typename... CallArgs>
		static Ret forward_device(Func func, Index3D i, CallArgs&&... args)
		{
			return func(i, std::forward<CallArgs>(args)...);
		}
		
		template<typename... CallArgs>
		static Ret forward_device(Func func, Index4D i, CallArgs&&... args)
		{
			return func(i, std::forward<CallArgs>(args)...);
		}
	};
	
	template<typename Func>
	struct ConditionalIndexForwarder<false, Func>
	{
		using Ret = typename return_type<Func>::type;
		
		template<typename... CallArgs>
		static Ret forward(Func func, Index1D, CallArgs&&... args)
		{
			return func(std::forward<CallArgs>(args)...);
		}
		
		template<typename... CallArgs>
		static Ret forward(Func func, Index2D, CallArgs&&... args)
		{
			return func(std::forward<CallArgs>(args)...);
		}
		
		
		template<typename... CallArgs>
		static Ret forward_device(Func func, Index1D, CallArgs&&... args)
		{
			return func(std::forward<CallArgs>(args)...);
		}
		
		template<typename... CallArgs>
		static Ret forward_device(Func func, Index2D, CallArgs&&... args)
		{
			return func(std::forward<CallArgs>(args)...);
		}
	};
	
}
