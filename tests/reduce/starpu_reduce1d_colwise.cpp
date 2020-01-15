#include <catch2/catch.hpp>

#include <skepu3/cluster/containers/matrix/matrix.hpp>
#include <skepu3/cluster/containers/vector/vector.hpp>
#include <skepu3/cluster/skeletons/reduce/reduce.hpp>

struct sum_fn
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

	static inline SKEPU_ATTRIBUTE_FORCE_INLINE int OMP(int a, int b)
	{
		return a + b;
	}

	static inline SKEPU_ATTRIBUTE_FORCE_INLINE int CPU(int a, int b)
	{
		return a + b;
	}
};

TEST_CASE("Column wise sum of matrix")
{
	auto sum = skepu::backend::Reduce1D<sum_fn, bool, void>(false);
	size_t constexpr N{900};
	auto expected = std::vector<int>(N);
	auto m = skepu::Matrix<int>(N,N);
	auto res = skepu::Vector<int>(N);

	m.flush();
	res.flush();
	for(size_t i(0); i < N; ++i)
	{
		auto first = i * N;
		for(size_t j(0); j < N; ++j)
		{
			m(i,j) = first + j;
			expected[j] += first + j;
		}
	}
	auto mt = m;
	mt.transpose(0);

	sum.setReduceMode(skepu::ReduceMode::ColWise);
	REQUIRE_NOTHROW(res = sum(res, m));

	res.flush();
	m.flush();
	mt.flush();
	for(size_t i(0); i < N; ++i)
	{
		REQUIRE(res(i) == expected[i]);
		for(size_t j(0); j < N; ++j)
			REQUIRE(m(i,j) == mt(j,i));
	}

}
