#pragma once
#ifndef SKEPU_STARPU_VECTOR_PARTITION_HPP
#define SKEPU_STARPU_VECTOR_PARTITION_HPP 1

#include <starpu_mpi.h>

#include "../partition.hpp"

namespace skepu {
namespace util {

template<typename T>
class vector_partition : private partition_base<T>
{
	typedef partition_base<T> base;
public:
	vector_partition() noexcept
	: base()
	{}

	vector_partition(size_t count) noexcept
	: base()
	{
		set_sizes(count);
		if(count)
			alloc_partitions();
	}

	vector_partition(vector_partition const & other) noexcept
	: vector_partition(other.m_size)
	{
		base::copy(other);
	}

	vector_partition(vector_partition && other) noexcept
	: base(std::move(other))
	{}

	~vector_partition() noexcept = default;

	auto
	operator=(vector_partition const & other) noexcept
	-> vector_partition &
	{
		this->~vector_partition();
		new(this) vector_partition(other);
		return *this;
	}

	auto
	operator=(vector_partition && other) noexcept
	-> vector_partition &
	{
		this->~vector_partition();
		new(this) vector_partition(std::move(other));
		return *this;
	}

	auto
	getParent() noexcept
	-> vector_partition &
	{
		return *this;
	}

	using base::operator();
	using base::allgather;
	using base::block_count_from;
	using base::capacity;
	using base::data;
	using base::fill;
	using base::local_storage_handle;
	using base::handle_for;
	using base::partition;
	using base::randomize;
	using base::set;
	using base::size;

private:
	auto
	alloc_local_storage() noexcept
	-> void override
	{
		base::m_data = new T[base::m_size];
		starpu_vector_data_register(
			&(base::m_data_handle),
			STARPU_MAIN_RAM,
			(uintptr_t)(base::m_data),
			base::m_size,
			sizeof(T));
		starpu_mpi_data_register(
			base::m_data_handle,
			cluster::mpi_tag(),
			STARPU_MAIN_RAM);
	}

	auto
	alloc_partitions() noexcept
	-> void override
	{
		if(base::m_size)
		{
			base::m_part_data = new T[base::m_part_size];
			for(size_t i(0); i < base::m_handles.size(); ++i)
			{
				if(i == cluster::mpi_rank())
					starpu_register(
						base::m_part_data,
						base::m_part_size,
						base::m_handles[i],
						i);
				else
					starpu_register(
						0, // The data is not on our node, so we don't need a ptr here.
						base::m_part_size,
						base::m_handles[i],
						i);
			}
		}
	}

	auto
	get_ptr(starpu_data_handle_t & handle) noexcept
	-> T * override
	{
		return (T *)starpu_vector_get_local_ptr(handle);
	}

	auto
	set_sizes(size_t count) noexcept
	-> void
	{
		base::m_size = count;
		if(count)
		{
			base::m_part_size = count / cluster::mpi_size();
			if(count % cluster::mpi_size())
				++base::m_part_size;
			base::m_capacity = base::m_part_size * cluster::mpi_size();
		}
	}

	auto
	starpu_register(
		T * ptr,
		size_t count,
		starpu_data_handle_t & handle,
		int rank) noexcept
	-> void
	{
		if(ptr)
			starpu_vector_data_register(
				&handle,
				STARPU_MAIN_RAM,
				(uintptr_t)ptr,
				count,
				sizeof(T));
		else
			starpu_vector_data_register(
				&handle,
				-1, // StarPU takes care of data management for other ranks handles.
				0,
				count,
				sizeof(T));

		starpu_mpi_data_register(
			handle,
			cluster::mpi_tag(),
			rank);
	}
};

} // namespace util
} // namespace skepu

#endif // SKEPU_STARPU_VECTOR_PARTITION_HPP
