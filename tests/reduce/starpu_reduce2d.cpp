#include <catch2/catch.hpp>

#include <skepu3/cluster/containers/matrix/matrix.hpp>
#include <skepu3/cluster/containers/vector/vector.hpp>
#include <skepu3/cluster/skeletons/reduce/reduce.hpp>

struct skepu_userfunction_red_mat_row_add
{
	constexpr static size_t totalArity = 2;
	constexpr static size_t outArity = 1;
	constexpr static bool indexed = 0;
	using ElwiseArgs = std::tuple<>;
	using ContainerArgs = std::tuple<>;
	using UniformArgs = std::tuple<int, int>;
	typedef std::tuple<> ProxyTags;
	constexpr static skepu::AccessMode anyAccessMode[] = {};
	using Ret = int;
	constexpr static bool prefersMatrix = 0;

	static inline int OMP(int a, int b)
	{
		return a + b;
	}

	static inline int CPU(int a, int b)
	{
		return a + b;
	}
};

struct skepu_userfunction_red_mat_col_mult
{
	constexpr static size_t totalArity = 2;
	constexpr static size_t outArity = 1;
	constexpr static bool indexed = 0;
	using ElwiseArgs = std::tuple<int, int>;
	using ContainerArgs = std::tuple<>;
	using UniformArgs = std::tuple<>;
	typedef std::tuple<> ProxyTags;
	constexpr static skepu::AccessMode anyAccessMode[] = {};
	using Ret = int;
	constexpr static bool prefersMatrix = 0;

	static inline int OMP(int a, int b)
	{
		return a * b;
	}

	static inline int CPU(int a, int b)
	{
		return a * b;
	}
};

TEST_CASE("Vector reduction")
{
	size_t constexpr N{10000};
	skepu::Vector<int> v(N);
	int expected(((N -1)*N)/2);
	auto sum =
		skepu::backend::Reduce2D<
				skepu_userfunction_red_mat_row_add,
				skepu_userfunction_red_mat_col_mult,
				bool, bool, void>
			(false,false);

	v.flush();
	for(size_t i(0); i < N; ++i)
		v(i) = i;

	int res;
	REQUIRE_NOTHROW(res = sum(v));
	CHECK(res == expected);
}

TEST_CASE("Matrix reduction")
{
	size_t constexpr N{16384};
	skepu::Matrix<int> m(N,N);
	int expected = 1;
	auto sum =
		skepu::backend::Reduce2D<
				skepu_userfunction_red_mat_row_add,
				skepu_userfunction_red_mat_col_mult,
				bool, bool, void>
			(false,false);

	m.flush();
	for(size_t i(0); i < N; ++i)
	{
		int row_res = 0;
		for(size_t j(0); j < N; ++j)
		{
			m(i,j) = i*j;
			row_res += i*j;
		}
		expected *= row_res;
	}

	int res;
	REQUIRE_NOTHROW(res = sum(m));
	CHECK(res == expected);
}
