#include <random>

#include <catch2/catch.hpp>

#include <skepu3/cluster/containers/matrix/matrix.hpp>
#include <skepu3/cluster/containers/vector/vector.hpp>
#include <skepu3/cluster/skeletons/map/mappairs.hpp>

std::mt19937 gen(1234);
std::uniform_int_distribution<int> dist(0,100);

struct mapp_index
{
	constexpr static size_t totalArity = 2;
	constexpr static size_t outArity = 1;
	constexpr static bool indexed = 1;
	using IndexType = skepu::Index2D;
	using ElwiseArgs = std::tuple<>;
	using ContainerArgs = std::tuple<>;
	using UniformArgs = std::tuple<unsigned long>;
	typedef std::tuple<> ProxyTags;
	constexpr static skepu::AccessMode anyAccessMode[] = {};
	using Ret = int;
	constexpr static bool prefersMatrix = 1;

	auto inline static
	OMP(skepu::Index2D i, unsigned long width)
	-> int
	{
		return (int)i.row * width + i.col;
	}

	auto inline static
	CPU(skepu::Index2D i, unsigned long width)
	-> int
	{
		return (int)i.row * width + i.col;
	}
};

TEST_CASE("Mapping with index")
{
	skepu::Matrix<int> m(4,2);
	auto mpi = skepu::backend::MapPairs<0, 0, mapp_index, bool, void>(false);

	REQUIRE_NOTHROW(mpi(m, m.size_j()));

	m.flush();
	for(size_t i(0); i < m.size(); ++i)
		REQUIRE((int)i == *(m.begin() + i));
}

struct vv_add
{
	constexpr static size_t totalArity = 2;
	constexpr static size_t outArity = 1;
	constexpr static bool indexed = 0;
	using IndexType = void;
	using ElwiseArgs = std::tuple<int, int>;
	using ContainerArgs = std::tuple<>;
	using UniformArgs = std::tuple<>;
	typedef std::tuple<> ProxyTags;
	constexpr static skepu::AccessMode anyAccessMode[] = {};
	using Ret = int;
	constexpr static bool prefersMatrix = 1;

	auto inline static
	OMP(int a, int b) noexcept
	-> int
	{
		return a + b;
	}

	auto inline static
	CPU(int a, int b) noexcept
	-> int
	{
		return a + b;
	}
};

TEST_CASE("Maping two vectors to matrix")
{
	REQUIRE_NOTHROW(
		skepu::backend::MapPairs<1,1, vv_add, bool, void>(false));
	auto map = skepu::backend::MapPairs<1,1, vv_add, bool, void>(false);

	SECTION("Size 0x0")
	{
		auto mat = skepu::Matrix<int>(0,0);
		auto v1 = skepu::Vector<int>(0);
		auto v2 = skepu::Vector<int>(0);

		REQUIRE_NOTHROW(map(mat,v1,v2));
	}

	SECTION("Size 1x1")
	{
		auto mat = skepu::Matrix<int>(1,1);
		auto v1 = skepu::Vector<int>(1,1);
		auto v2 = skepu::Vector<int>(1,2);

		REQUIRE_NOTHROW(map(mat,v1,v2));
		mat.flush();
		CHECK(mat(0,0) == 3);
	}

	SECTION("Size 2x2")
	{
		auto mat = skepu::Matrix<int>(2,2);
		auto v1 = skepu::Vector<int>(2);
		auto v2 = skepu::Vector<int>(2);

		v1.flush();
		v2.flush();
		v1(0) = 1;
		v1(1) = 2;
		v2(0) = 3;
		v2(1) = 4;

		REQUIRE_NOTHROW(map(mat,v1,v2));
		mat.flush();
		CHECK(mat(0,0) == 4);
		CHECK(mat(0,1) == 5);
		CHECK(mat(1,0) == 5);
		CHECK(mat(1,1) == 6);
	}

	SECTION("size 100x100")
	{
		auto mat = skepu::Matrix<int>(100,100);
		auto expected = skepu::Matrix<int>(100,100);
		auto v1 = skepu::Vector<int>(100);
		auto v2 = skepu::Vector<int>(100);

		v1.flush();
		v2.flush();
		for(size_t i(0); i < 100; ++i)
		{
			v1(i) = dist(gen);
			v2(i) = dist(gen);
		}

		expected.flush();
		for(size_t i(0); i < 100; ++i)
			for(size_t j(0); j < 100; ++j)
				expected(i,j) = v1(i) + v2(j);

		REQUIRE_NOTHROW(map(mat, v1,v2));
		mat.flush();
		auto ex_it = expected.begin();
		for(auto it = mat.begin(); it != mat.end(); ++it, ++ex_it)
			REQUIRE(*it == *ex_it);
	}
}
