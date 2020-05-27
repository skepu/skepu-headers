#include <catch2/catch.hpp>

#include <skepu3/cluster/containers/vector/vector.hpp>
#include <skepu3/cluster/skeletons/mapreduce.hpp>

struct scalar_fn
{
	constexpr static size_t totalArity = 1;
	constexpr static size_t outArity = 1;
	constexpr static bool indexed = 0;
	using IndexType = void;
	using ElwiseArgs = std::tuple<>;
	using ContainerArgs = std::tuple<>;
	using UniformArgs = std::tuple<int>;
	typedef std::tuple<> ProxyTags;
	constexpr static skepu::AccessMode anyAccessMode[] = {};
	using Ret = int;
	constexpr static bool prefersMatrix = 0;

	auto static inline
	OMP(int i) noexcept
	-> int
	{
		return i;
	}

	auto static inline
	CPU(int i) noexcept
	-> int
	{
		return i;
	}
};

struct add_int
{
	constexpr static size_t totalArity = 2;
	constexpr static size_t outArity = 1;
	constexpr static bool indexed = 0;
	using IndexType = void;
	using ElwiseArgs = std::tuple<>;
	using ContainerArgs = std::tuple<>;
	using UniformArgs = std::tuple<int, int>;
	typedef std::tuple<> ProxyTags;
	constexpr static skepu::AccessMode anyAccessMode[] = {};
	using Ret = int;
	constexpr static bool prefersMatrix = 0;

	auto static inline
	OMP(int a, int b) noexcept
	-> int
	{
		return a + b;
	}

	auto static inline
	CPU(int a, int b) noexcept
	-> int
	{
		return a + b;
	}
};

TEST_CASE("1D Scalar MapReduce")
{
	REQUIRE_NOTHROW(
		skepu::backend::MapReduce<0, scalar_fn, add_int, bool, bool, void>(
			false, false));
	auto map_red =
		skepu::backend::MapReduce<0, scalar_fn, add_int, bool, bool, void>(
			false, false);
	int static constexpr value = 10;

	SECTION("1D size 0")
	{
		map_red.setDefaultSize(0);
		auto res = map_red(value);
		CHECK(res == 0);
	}

	SECTION("1D size 1")
	{
		map_red.setDefaultSize(1);
		auto res = map_red(value);
		CHECK(res == 10);
	}

	SECTION("1D size 2")
	{
		map_red.setDefaultSize(2);
		auto res = map_red(value);
		CHECK(res == 20);
	}

	SECTION("1D size 1000")
	{
		map_red.setDefaultSize(1000);
		auto res = map_red(value);
		CHECK(res == 10000);
	}
}

struct copy_fn
{
	constexpr static size_t totalArity = 1;
	constexpr static size_t outArity = 1;
	constexpr static bool indexed = 0;
	using IndexType = void;
	using ElwiseArgs = std::tuple<int>;
	using ContainerArgs = std::tuple<>;
	using UniformArgs = std::tuple<>;
	typedef std::tuple<> ProxyTags;
	constexpr static skepu::AccessMode anyAccessMode[] = {};
	using Ret = int;
	constexpr static bool prefersMatrix = 0;

	auto static inline
	OMP(int i) noexcept
	-> int
	{
		return i;
	}

	auto static inline
	CPU(int i) noexcept
	-> int
	{
		return i;
	}
};

TEST_CASE("Elementwise Vector MapReduce")
{
	REQUIRE_NOTHROW(
		skepu::backend::MapReduce<
			1, copy_fn, add_int, bool, bool, void>(false, false));
	auto mapreduce =	skepu::backend::MapReduce<
			1, copy_fn, add_int, bool, bool, void>(false, false);

	SECTION("vector size 1")
	{
		auto v = skepu::Vector<int>(1, 0);
		int expected = 0;
		auto res = mapreduce(v);
		CHECK(res == expected);
	}

	SECTION("vector size 2")
	{
		auto v = skepu::Vector<int>(2);
		v.flush();
		v(0) = 0;
		v(1) = 1;
		int expected = 1;
		auto res = mapreduce(v);
		CHECK(res == expected);
	}

	SECTION("vector size 10")
	{
		auto v = skepu::Vector<int>(10);
		int expected = 45;
		v.flush();
		for(size_t i(0); i < v.size(); ++i)
			v(i) = i;
		auto res = mapreduce(v);
		CHECK(res == expected);
	}

	SECTION("vector size 1000")
	{
		size_t constexpr N(1000);
		auto v = skepu::Vector<int>(N);
		int constexpr expected = ((N-1)*N)/2;
		v.flush();
		for(size_t i(0); i < v.size(); ++i)
			v(i) = i;
		auto res = mapreduce(v);
		CHECK(res == expected);
	}

	SECTION("with iterators")
	{
		size_t constexpr N(1000);
		auto v = skepu::Vector<int>(N);
		int constexpr expected = ((N-1)*N)/2;
		v.flush();
		for(size_t i(0); i < v.size(); ++i)
			v(i) = i;
		auto res = mapreduce(v.begin(), v.end());
		CHECK(res == expected);
	}
}
