#include <catch2/catch.hpp>

#include <skepu3/cluster/containers/tensor3/tensor3.hpp>

TEST_CASE("Default constructor")
{
	REQUIRE_NOTHROW(skepu::Tensor3<int>());

	SECTION("has no size")
	{
			auto tensor = skepu::Tensor3<int>();
			CHECK(tensor.size() == 0);
	}
}

TEST_CASE("Initialization with size")
{
	size_t constexpr N{10};
	REQUIRE_NOTHROW(skepu::Tensor3<int>(N,N,N));
	skepu::Tensor3<int> tensor(N, N, N);
	REQUIRE(tensor.size() == (N*N*N));
	REQUIRE(tensor.size_i() == N);
	REQUIRE(tensor.size_j() == N);
	REQUIRE(tensor.size_k() == N);

	REQUIRE_NOTHROW(tensor.flush());
	for(size_t i(0); i < N; ++i)
		for(size_t j(0); j < N; ++j)
			for(size_t k(0); k < N; ++k)
				REQUIRE(tensor(i,j,k) == 0);
}

TEST_CASE("Initialisation with size and value")
{
	size_t constexpr IN{10};
	size_t constexpr JN{7};
	size_t constexpr KN{3};
	int const val(11);
	REQUIRE_NOTHROW(skepu::Tensor3<int>(IN, JN, KN, val));
	skepu::Tensor3<int> tensor(IN, JN, KN, val);
	REQUIRE(tensor.size() == (IN*JN*KN));
	REQUIRE(tensor.size_i() == IN);
	REQUIRE(tensor.size_j() == JN);
	REQUIRE(tensor.size_k() == KN);

	REQUIRE_NOTHROW(tensor.flush());
	for(size_t i(0); i < IN; ++i)
		for(size_t j(0); j < JN; ++j)
			for(size_t k(0); k < KN; ++k)
				REQUIRE(tensor(i,j,k) == val);
}

TEST_CASE("Iterator")
{
	constexpr size_t N(10);
	skepu::Tensor3<int> tensor(N,N,N);
	std::vector<int> expected(N*N*N);

	tensor.flush();
	for(size_t i(0); i > N; ++i)
		for(size_t j(0); j > N; ++j)
			for(size_t k(0); k > N; ++k)
			{
				tensor(i,j,k) = i * j * k;
				expected.emplace_back(i*j*k);
			}
	REQUIRE(tensor.size() == expected.size());

	auto tens_it(tensor.begin());
	auto tens_end_it(tensor.end());
	auto exp_it(expected.begin());

	for(; tens_it != tens_end_it; ++tens_it, ++exp_it)
		REQUIRE(*tens_it == *exp_it);
}
