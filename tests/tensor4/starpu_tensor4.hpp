#include <catch2/catch.hpp>

#include <skepu3/cluster/containers/tensor4/tensor4.hpp>

TEST_CASE("Default constructor")
{
	REQUIRE_NOTHROW(skepu::Tensor4<int>());

	SECTION("has no size")
	{
			auto tensor = skepu::Tensor4<int>();
			CHECK(tensor.size() == 0);
	}
}

TEST_CASE("Initialization with size")
{
	size_t constexpr I{1};
	size_t constexpr J{2};
	size_t constexpr K{3};
	size_t constexpr L{5};
	REQUIRE_NOTHROW(skepu::Tensor4<int>(I,J,K,L));
	skepu::Tensor4<int> tensor(I,J,K,L);
	REQUIRE(tensor.size() == (I*J*K*L));
	REQUIRE(tensor.size_i() == I);
	REQUIRE(tensor.size_j() == J);
	REQUIRE(tensor.size_k() == K);
	REQUIRE(tensor.size_l() == L);

	REQUIRE_NOTHROW(tensor.flush());
	for(size_t i(0); i < I; ++i)
		for(size_t j(0); j < J; ++j)
			for(size_t k(0); k < K; ++k)
				for(size_t l(0); l < L; ++l)
					REQUIRE(tensor(i,j,k,l) == 0);
}

TEST_CASE("Initialisation with size and value")
{
	size_t constexpr I{10};
	size_t constexpr J{7};
	size_t constexpr K{3};
	size_t constexpr L{3};
	int const val(11);
	REQUIRE_NOTHROW(skepu::Tensor4<int>(I, J, K, L, val));
	skepu::Tensor4<int> tensor(I, J, K, L, val);
	REQUIRE(tensor.size() == (I*J*K*L));
	REQUIRE(tensor.size_i() == I);
	REQUIRE(tensor.size_j() == J);
	REQUIRE(tensor.size_k() == K);
	REQUIRE(tensor.size_l() == L);

	REQUIRE_NOTHROW(tensor.flush());
	for(size_t i(0); i < I; ++i)
		for(size_t j(0); j < J; ++j)
			for(size_t k(0); k < K; ++k)
				for(size_t l(0); l < L; ++l)
					REQUIRE(tensor(i,j,k,l) == val);
}

TEST_CASE("Iterator")
{
	constexpr size_t N(10);
	skepu::Tensor4<int> tensor(N,N,N,N);
	std::vector<int> expected(N*N*N*N);

	tensor.flush();
	for(size_t i(0); i < N; ++i)
		for(size_t j(0); j < N; ++j)
			for(size_t k(0); k < N; ++k)
				for(size_t l(0); l < N; ++l)
				{
					tensor(i,j,k,l) = i*j*k*l;
					expected.emplace_back(i*j*k*l);
				}
	REQUIRE(tensor.size() == expected.size());

	auto tens_it(tensor.begin());
	auto tens_end_it(tensor.end());
	auto exp_it(expected.begin());

	for(; tens_it != tens_end_it; ++tens_it, ++exp_it)
		REQUIRE(*tens_it == *exp_it);
}
