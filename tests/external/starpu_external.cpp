#include <random>

#include <catch2/catch.hpp>

#include <skepu3/cluster/external.hpp>
#include <skepu3/cluster/containers/matrix/matrix.hpp>
#include <skepu3/cluster/containers/tensor3/tensor3.hpp>
#include <skepu3/cluster/containers/vector/vector.hpp>

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<int> dist;

TEST_CASE("Only rank 0 calls operator")
{
	bool reched_operator(false);

	REQUIRE_NOTHROW(
		skepu::external([&]
		{
			reched_operator = true;
		}));

	if(!skepu::cluster::mpi_rank())
		CHECK(reched_operator);
	else
		CHECK(!reched_operator);
}

template<typename T>
struct container_stub
{
	bool flushed;
	bool partitioned;

	void gather_to_root() { flushed = true; }
	void scatter_from_root() { partitioned = true; }

	auto inline
	getParent() noexcept
	-> container_stub &
	{
		return *this;
	}
};

namespace skepu {

template<>
struct is_skepu_container<container_stub<int> &> : public std::true_type
{};

} // namespace skepu

TEST_CASE("One container is flushed and partitioned.")
{
	container_stub<int> c{false,false};
	
	REQUIRE_NOTHROW(
		skepu::external(c, [&]
		{
		}));

	CHECK(c.flushed);
	CHECK(c.partitioned);
}

TEST_CASE("Two containers are flushed and partitioned:")
{
	container_stub<int> c1{false,false};
	container_stub<int> c2{false,false};
	
	REQUIRE_NOTHROW(
		skepu::external(c1, c2, [&]
		{
		}));

	CHECK(c1.flushed);
	CHECK(c1.partitioned);
	CHECK(c2.flushed);
	CHECK(c2.partitioned);
}

TEST_CASE("Vector is gathered before operator call")
{
	size_t constexpr N{10};
	auto expected = std::vector<int>(N * skepu::cluster::mpi_size());
	auto local = std::vector<int>(N);
	for(auto & e : local)
		e = dist(gen);
	MPI_Gather(
		local.data(), N, MPI_INT,
		expected.data(), N, MPI_INT,
		0, MPI_COMM_WORLD);
	skepu::Vector<int> v(N * skepu::cluster::mpi_size());
	auto & partition = skepu::cont::getParent(v);
	auto handle = partition.handle_for(N * skepu::cluster::mpi_rank());
	starpu_data_acquire(handle, STARPU_RW);
	auto part_it = (int *)starpu_vector_get_local_ptr(handle);
	std::copy(local.begin(), local.end(), part_it);
	starpu_data_release(handle);

	skepu::external(v, [&]
	{
		auto v_it = v.begin();
		auto v_end = v.end();
		auto e_it = expected.begin();
		for(; v_it != v_end; ++v_it, ++e_it)
			CHECK(*v_it == *e_it);
	});
}

TEST_CASE("Vector is scattered after the operator call")
{
	size_t constexpr N{10};
	auto init_v = std::vector<int>(N * skepu::cluster::mpi_size());
	auto expected = std::vector<int>(N);
	for(auto & e : init_v)
		e = dist(gen);
	MPI_Scatter(
		init_v.data(), N, MPI_INT,
		expected.data(), N, MPI_INT,
		0, MPI_COMM_WORLD);

	skepu::Vector<int> v(N * skepu::cluster::mpi_size());
	skepu::external(v, [&]
	{
		auto v_it = v.begin();
		auto v_end = v.end();
		auto i_it = init_v.begin();
		for(; v_it != v_end; ++v_it, ++i_it)
			*v_it = *i_it;
	});

	auto & partition = skepu::cont::getParent(v);
	auto handle = partition.handle_for(skepu::cluster::mpi_rank() * N);
	starpu_data_acquire(handle, STARPU_R);
	auto it = (int *)starpu_vector_get_local_ptr(handle);
	for(size_t i(0); i < N; ++i)
		CHECK(it[i] == expected[i]);
	starpu_data_release(handle);
}

TEST_CASE("Matrix is gathered before operator call")
{
	size_t constexpr R{10};
	size_t constexpr C{10};
	size_t constexpr N{R*C};
	auto expected = std::vector<int>(N * skepu::cluster::mpi_size());
	auto local = std::vector<int>(N);
	for(auto & e : local)
		e = dist(gen);
	MPI_Gather(
		local.data(), N, MPI_INT,
		expected.data(), N, MPI_INT,
		0, MPI_COMM_WORLD);
	skepu::Matrix<int> m(R * skepu::cluster::mpi_size(), C);
	auto & partition = skepu::cont::getParent(m);
	auto handle = partition.handle_for(N * skepu::cluster::mpi_rank());
	starpu_data_acquire(handle, STARPU_RW);
	auto part_it = (int *)starpu_matrix_get_local_ptr(handle);
	std::copy(local.begin(), local.end(), part_it);
	starpu_data_release(handle);

	skepu::external(m, [&]
	{
		auto v_it = m.begin();
		auto v_end = m.end();
		auto e_it = expected.begin();
		for(; v_it != v_end; ++v_it, ++e_it)
			CHECK(*v_it == *e_it);
	});
}

TEST_CASE("Matrix is scattered after the operator call")
{
	size_t constexpr R{10};
	size_t constexpr C{10};
	size_t constexpr N{R*C};
	auto init_v = std::vector<int>(N * skepu::cluster::mpi_size());
	auto expected = std::vector<int>(N);
	for(auto & e : init_v)
		e = dist(gen);
	MPI_Scatter(
		init_v.data(), N, MPI_INT,
		expected.data(), N, MPI_INT,
		0, MPI_COMM_WORLD);

	skepu::Matrix<int> m(R * skepu::cluster::mpi_size(), C);
	skepu::external(m, [&]
	{
		auto v_it = m.begin();
		auto v_end = m.end();
		auto i_it = init_v.begin();
		for(; v_it != v_end; ++v_it, ++i_it)
			*v_it = *i_it;
	});

	auto & partition = skepu::cont::getParent(m);
	auto handle = partition.handle_for(skepu::cluster::mpi_rank() * N);
	starpu_data_acquire(handle, STARPU_R);
	auto it = (int *)starpu_matrix_get_local_ptr(handle);
	for(size_t i(0); i < N; ++i)
		CHECK(it[i] == expected[i]);
	starpu_data_release(handle);
}

TEST_CASE("Tensor3 is gathered before operator call")
{
	size_t constexpr I{10};
	size_t constexpr J{10};
	size_t constexpr K{10};
	size_t constexpr N{I*J*K};
	auto expected = std::vector<int>(N * skepu::cluster::mpi_size());
	auto local = std::vector<int>(N);
	for(auto & e : local)
		e = dist(gen);
	MPI_Gather(
		local.data(), N, MPI_INT,
		expected.data(), N, MPI_INT,
		0, MPI_COMM_WORLD);
	skepu::Tensor3<int> t3(I * skepu::cluster::mpi_size(), J, K);
	auto & partition = skepu::cont::getParent(t3);
	/*
	int trap(1);
	while(trap);
	*/
	auto handle = partition.handle_for(N * skepu::cluster::mpi_rank());
	starpu_data_acquire(handle, STARPU_RW);
	auto part_it = (int *)starpu_block_get_local_ptr(handle);
	std::copy(local.begin(), local.end(), part_it);
	starpu_data_release(handle);

	skepu::external(t3, [&]
	{
		auto v_it = t3.begin();
		auto v_end = t3.end();
		auto e_it = expected.begin();
		for(; v_it != v_end; ++v_it, ++e_it)
			CHECK(*v_it == *e_it);
	});
}

/*
TEST_CASE("Tensor3 is scattered after the operator call")
{
	size_t constexpr I{10};
	size_t constexpr J{10};
	size_t constexpr K{10};
	size_t constexpr N{I*J*K};
	auto init_v = std::vector<int>(N * skepu::cluster::mpi_size());
	auto expected = std::vector<int>(N);
	for(auto & e : init_v)
		e = dist(gen);
	MPI_Scatter(
		init_v.data(), N, MPI_INT,
		expected.data(), N, MPI_INT,
		0, MPI_COMM_WORLD);

	skepu::Tensor3<int> t3(I * skepu::cluster::mpi_size(), J, K);
	skepu::external(t3, [&]
	{
		auto v_it = t3.begin();
		auto v_end = t3.end();
		auto i_it = init_v.begin();
		for(; v_it != v_end; ++v_it, ++i_it)
			*v_it = *i_it;
	});

	auto & partition = skepu::cont::getParent(t3);
	auto handle = partition.handle_for(skepu::cluster::mpi_rank() * N);
	starpu_data_acquire(handle, STARPU_R);
	auto it = (int *)starpu_block_get_local_ptr(handle);
	for(size_t i(0); i < N; ++i)
		CHECK(it[i] == expected[i]);
	starpu_data_release(handle);
}
*/
