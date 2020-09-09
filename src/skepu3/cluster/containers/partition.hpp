#pragma once
#ifndef SKEPU_STARPU_PARTITION_BASE_HPP
#define SKEPU_STARPU_PARTITION_BASE_HPP 1

#include <random>
#include <vector>

#include "../cluster.hpp"

namespace skepu {

template<typename T>
class partition_base
{
protected:
	T * m_data;
	T * m_part_data;

	bool m_data_valid;
	bool m_part_valid;

	size_t m_size;
	size_t m_part_size;
	size_t m_capacity;

	starpu_data_handle_t m_data_handle;
	std::vector<starpu_data_handle_t> m_handles;

	bool m_external;

	partition_base() noexcept
	: m_data(0),
		m_part_data(0),
		m_data_valid(false),
		m_part_valid(false),
		m_size(0),
		m_part_size(0),
		m_capacity(0),
		m_data_handle(0),
		m_handles(cluster::mpi_size(), 0),
		m_external(false)
	{}

	partition_base(partition_base && other) noexcept
	: m_data(std::move(other.m_data)),
		m_part_data(std::move(other.m_part_data)),
		m_data_valid(std::move(other.m_data_valid)),
		m_part_valid(std::move(other.m_part_valid)),
		m_size(std::move(other.m_size)),
		m_part_size(std::move(other.m_part_size)),
		m_capacity(std::move(other.m_capacity)),
		m_data_handle(std::move(other.m_data_handle)),
		m_handles(std::move(other.m_handles)),
		m_external(false)
	{
		other.m_data = 0;
		other.m_part_data = 0;
	}

	~partition_base() noexcept
	{
		dealloc_local_storage();
		dealloc_partitions();
	}

	auto
	copy(partition_base const & other) noexcept
	-> void
	{
		if(other.m_part_valid)
		{
			starpu_data_acquire(other.m_handles[cluster::mpi_rank()], STARPU_R);
			starpu_data_acquire(m_handles[cluster::mpi_rank()], STARPU_W);
			std::copy(
				other.m_part_data, other.m_part_data + m_part_size, m_part_data);
			starpu_data_release(other.m_handles[cluster::mpi_rank()]);
			starpu_data_release(m_handles[cluster::mpi_rank()]);
			m_part_valid = true;
		}
		if(other.m_data_valid)
		{
			if(!m_data)
				alloc_local_storage();

			starpu_data_acquire(other.m_data_handle, STARPU_R);
			starpu_data_acquire(m_data_handle, STARPU_W);
			std::copy(other.m_data, other.m_data + m_size, m_data);
			starpu_data_release(m_data_handle);
			starpu_data_release(other.m_data_handle);
			m_data_valid = true;
		}
	}

	auto
	operator()(size_t pos) noexcept
	-> T &
	{
		if(m_data && m_data_valid)
			m_part_valid = false;
		return m_data[pos];
	}

	auto
	operator()(size_t pos) const noexcept
	-> T const &
	{
		return m_data[pos];
	}

	auto
	fill(size_t count, T const & value) noexcept
	-> void
	{
		if(m_external)
		{
			std::fill(m_data, m_data + m_size, value);
			return;
		}
		if(m_size < m_part_size * cluster::mpi_rank())
			return;

		if(!m_part_valid)
			partition();

		auto handle = m_handles[cluster::mpi_rank()];
		auto size =
			cluster::mpi_rank() == cluster::mpi_size() -1
			? m_size - (m_part_size * cluster::mpi_rank())
			: m_part_size;

		starpu_data_acquire(handle, STARPU_W);

		std::fill(m_part_data, m_part_data + size, value);

		starpu_data_release(handle);

/*
		auto end_part_idx = count / m_part_size;
		auto end_idx = count % m_part_size;
		if(end_part_idx >= cluster::mpi_rank())
		{
			starpu_data_acquire(m_handles[cluster::mpi_rank()], STARPU_W);
			if(end_part_idx > cluster::mpi_rank())
				std::fill(m_part_data, m_part_data + m_part_size, value);
			else
				std::fill(m_part_data, m_part_data + end_idx, value);
			starpu_data_release(m_handles[cluster::mpi_rank()]);
		}
*/

		m_data_valid = false;
		m_part_valid = true;
	}

	auto
	set(size_t pos, T const & value) noexcept
	-> void
	{
		if(!m_part_valid)
			partition();

		auto part_idx = pos / m_part_size;
		auto & handle = m_handles[part_idx];
		if(starpu_mpi_data_get_rank(handle) == cluster::mpi_rank())
		{
			auto local_idx = pos % m_part_size;
			starpu_data_acquire(handle, STARPU_W);
			m_part_data[local_idx] = value;
			starpu_data_release(handle);
		}

		m_data_valid = false;
		m_part_valid = true;
	}

	template<typename RandomIter>
	auto
	set(RandomIter begin, RandomIter end) noexcept
	-> void
	{
		if(!m_part_valid)
			partition();

		m_size = end - begin;
		auto end_part_idx = m_size / m_part_size;
		if(end_part_idx >= cluster::mpi_rank())
		{
			begin += cluster::mpi_rank() * m_part_size;
			auto & handle = m_handles[cluster::mpi_rank()];
			if(end_part_idx > cluster::mpi_rank())
				end = begin + m_part_size;

			starpu_data_acquire(handle, STARPU_W);
			std::copy(begin, end, m_part_data);
			starpu_data_release(handle);
		}

		m_data_valid = false;
		m_part_valid = true;
	}

	auto
	size() const noexcept
	-> size_t const &
	{
		return m_size;
	}

public:
	auto
	allgather() noexcept
	-> void
	{
		if(m_data_valid || m_external)
			return;

		if(m_capacity)
		{
			if(!m_data)
				alloc_local_storage();

			if(m_size)
			{
				auto part_end_idx = (m_size -1) / m_part_size;
				starpu_data_acquire(m_data_handle, STARPU_W);
				auto data_it = m_data;
				for(size_t i(0); i < part_end_idx; ++i)
					data_it = gather(m_handles[i], m_part_size, data_it);
				auto last_part_count = m_size - (part_end_idx * m_part_size);
				gather(m_handles[part_end_idx], last_part_count, data_it);
				starpu_data_release(m_data_handle);
			}
		}

		m_data_valid = true;
	}

	auto
	block_count_from(size_t pos) const noexcept
	-> size_t
	{
		auto block_idx = pos/m_part_size;
		size_t block_count{
			block_idx == cluster::mpi_size() -1
			? block_count = m_size - pos
			: block_count = m_part_size - (pos - (block_idx * m_part_size))};
		return block_count;
	}

	auto
	capacity() const noexcept
	-> size_t
	{
		return m_capacity;
	}

	auto
	data() noexcept
	-> T *
	{
		if(m_data && m_data_valid)
			m_part_valid = false;
		return m_data;
	}

	auto
	gather_to_root() noexcept
	-> void
	{
		if(m_data_valid)
			return;

		if(m_capacity)
		{
			if(!m_data)
				alloc_local_storage();

			if(m_size)
			{
				auto rank = cluster::mpi_rank();
				size_t end = m_size / m_part_size;
				if(!rank)
					starpu_data_acquire(m_data_handle, STARPU_W);

				auto data_it = m_data;
				if(!rank)
				{
					starpu_data_acquire(m_handles.front(), STARPU_R);
					size_t count = std::min(m_part_size, m_size);
					data_it = std::copy(m_part_data, m_part_data + count, data_it);
					starpu_data_release(m_handles.front());
				}
				for(size_t i(1); i < end; ++i)
				{
					MPI_Status recv_stat;
					auto tag = cluster::mpi_tag();
					if(i == rank)
						starpu_mpi_send(m_handles[i], 0, tag, MPI_COMM_WORLD);
					else if(!rank)
					{
						starpu_mpi_recv(m_handles[i], i, tag, MPI_COMM_WORLD, &recv_stat);
						size_t count = std::min(m_part_size, m_size - (m_part_size * i));
						starpu_data_acquire(m_handles[i], STARPU_R);
						auto part_it = get_ptr(m_handles[i]);
						data_it =
							std::copy(part_it, part_it + count, data_it);
						starpu_data_release(m_handles[i]);
						starpu_mpi_cache_flush(MPI_COMM_WORLD, m_handles[i]);
					}
				}

				if(!rank)
					starpu_data_release(m_data_handle);
			}
		}
	}

	auto
	handle_for(size_t pos) noexcept
	-> starpu_data_handle_t
	{
		m_data_valid = false;
		return m_handles[pos/m_part_size];
	}

	auto
	local_storage_handle() noexcept
	-> starpu_data_handle_t
	{
		m_part_valid = false;
		return m_data_handle;
	}

	auto
	make_ext_w() noexcept
	-> void
	{
		m_external = true;
		if(m_size)
			alloc_local_storage();
	}

	auto
	partition() noexcept
	-> void
	{
		if(m_part_valid)
			return;

		if(m_size && m_data)
		{
			auto rank = cluster::mpi_rank();
			auto end_rank = (m_size -1) / m_part_size;

			/* Only ranks that have data should copy anything. */
			if(rank <= end_rank)
			{
				starpu_data_acquire(m_data_handle, STARPU_R);
				auto begin = m_data + (rank * m_part_size);
				auto end =
					rank == end_rank
					? begin + (m_size - (rank * m_part_size))
					: begin + m_part_size;

				starpu_data_acquire(m_handles[rank], STARPU_W);
				std::copy(begin, end, m_part_data);
				starpu_data_release(m_handles[rank]);
				starpu_data_release(m_data_handle);
			}
		}

		m_part_valid = true;
	}

	auto
	scatter_from_root() noexcept
	-> void
	{
		m_external = false;
		update_sizes();

		if(m_data)
		{
			auto rank = cluster::mpi_rank();
			auto data_it = m_data;

			if(!rank)
			{
				starpu_data_acquire(m_data_handle, STARPU_R);
				starpu_data_acquire(m_handles.front(), STARPU_W);
				auto count = std::min(m_part_size, m_size);
				std::copy(data_it, data_it + count, m_part_data);
				starpu_data_release(m_handles.front());
				data_it += count;
			}
			for(size_t i(1); i < cluster::mpi_size(); ++i)
			{
				auto tag = cluster::mpi_tag();
				auto & handle = m_handles[i];

				if(!rank)
				{
					starpu_data_acquire(handle, STARPU_W);
					auto count = std::min(m_part_size, m_size - (i * m_part_size));
					auto out_it = get_ptr(handle);
					std::copy(data_it, data_it + count, out_it);
					starpu_data_release(handle);
					data_it += count;
					starpu_mpi_send(handle, i, tag, MPI_COMM_WORLD);
				}
				else if(i == rank)
				{
					MPI_Status status;
					starpu_mpi_recv(handle, 0, tag, MPI_COMM_WORLD, &status);
				}
				starpu_mpi_cache_flush(MPI_COMM_WORLD, handle);
			}

			if(!rank)
				starpu_data_release(m_data_handle);
		}

		m_part_valid = true;
		m_data_valid = false;
	}

	template<typename U,
		typename std::enable_if<std::is_integral<U>::value, int>::type = 0>
	auto
	randomize(U const & min, U const & max) noexcept
	-> void
	{
		std::random_device rd;
		std::mt19937_64 generator(rd());
		std::uniform_int_distribution<U> distributor(min, max);

		auto rank = cluster::mpi_rank();
		auto start_idx = rank * m_part_size;
		if(start_idx < m_size)
		{
			auto start_it = m_part_data;
			auto end_it = start_it + std::min(m_part_size, m_size - start_idx);

			auto handle = m_handles[cluster::mpi_rank()];
			starpu_data_acquire(handle, STARPU_W);
			for(; start_it != end_it; ++start_it)
				*start_it = distributor(generator);
			starpu_data_release(handle);
		}

		m_part_valid = true;
		m_data_valid = false;
	}

	template<typename U,
		typename std::enable_if<std::is_floating_point<U>::value, int>::type = 0>
	auto
	randomize(U const & min, U const & max) noexcept
	-> void
	{
		std::random_device rd;
		std::mt19937_64 generator(rd());
		std::uniform_real_distribution<U> distributor(min, max);

		auto rank = cluster::mpi_rank();
		auto start_idx = rank * m_part_size;
		if(start_idx < m_size)
		{
			auto start_it = m_part_data;
			auto end_it = start_it + std::min(m_part_size, m_size - start_idx);

			auto handle = m_handles[cluster::mpi_rank()];
			starpu_data_acquire(handle, STARPU_W);
			for(; start_it != end_it; ++start_it)
				*start_it = distributor(generator);
			starpu_data_release(handle);
		}

		m_part_valid = true;
		m_data_valid = false;
	}

private:
	virtual auto alloc_local_storage() noexcept -> void = 0;
	virtual auto alloc_partitions() noexcept -> void = 0;
	virtual auto get_ptr(starpu_data_handle_t & handle) noexcept -> T * = 0;
	virtual auto update_sizes() noexcept -> void = 0;

	auto
	dealloc_local_storage()
	-> void
	{
		if(m_data)
		{
			starpu_data_acquire(m_data_handle, STARPU_RW);
			starpu_data_release(m_data_handle);
			starpu_data_unregister_no_coherency(m_data_handle);
			delete[] m_data;
			m_data = 0;
		}
	}

	auto
	dealloc_partitions() noexcept
	-> void
	{
		if(m_part_data)
		{
			for(auto & handle : m_handles)
			{
				starpu_data_unregister_no_coherency(handle);
			}
			delete[] m_part_data;
			m_part_data = 0;
		}
	}

	template<typename Iterator>
	auto
	gather(starpu_data_handle_t handle, size_t count, Iterator data_it) noexcept
	-> Iterator
	{
		starpu_mpi_get_data_on_all_nodes_detached(MPI_COMM_WORLD, handle);
		starpu_data_acquire(handle, STARPU_R);
		auto part_it = (T *)get_ptr(handle);
		data_it = std::copy(part_it, part_it + count, data_it);
		starpu_data_release(handle);
		starpu_mpi_cache_flush(MPI_COMM_WORLD, handle);

		return data_it;
	}
};

} // namespace skepu

#endif // SKEPU_STARPU_PARTITION_BASE_HPP
