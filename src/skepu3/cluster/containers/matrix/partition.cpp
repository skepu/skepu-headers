#pragma once
#ifndef SKEPU_STARPU_MATRIX_PARTITION_HPP
#define SKEPU_STARPU_MATRIX_PARTITION_HPP 1

#include "../partition.hpp"

namespace skepu {
namespace util {

template<typename T>
class matrix_partition : private partition_base<T>
{
	typedef partition_base<T> base;
	size_t m_rows;
	size_t m_cols;
	size_t m_part_rows;

public:
	matrix_partition() noexcept
	: base(), m_rows(0), m_cols(0), m_part_rows(0)
	{}

	matrix_partition(size_t rows, size_t cols) noexcept
	: base(), m_part_rows(0)
	{
		init(rows, cols);
	}

	matrix_partition(matrix_partition const & other) noexcept
	: base(),
		m_rows(other.m_rows),
		m_cols(other.m_cols),
		m_part_rows(other.m_part_rows)
	{
		set_sizes();
		if(base::m_size)
		{
			alloc_partitions();
			base::copy(other);
		}
	}

	matrix_partition(matrix_partition && other) noexcept
	: base(std::move(other)),
		m_rows(std::move(other.m_rows)),
		m_cols(std::move(other.m_cols)),
		m_part_rows(std::move(other.m_part_rows))
	{}

	~matrix_partition() noexcept = default;

	auto
	init(size_t rows, size_t cols) noexcept
	-> void
	{
		m_rows = rows;
		m_cols = cols;

		set_sizes();
		if(base::m_external)
			alloc_local_storage();
		else
			alloc_partitions();
	}

	auto
	operator=(matrix_partition const & other) noexcept
	-> matrix_partition &
	{
		this->~matrix_partition();
		new(this) matrix_partition(other);
		return *this;
	}

	auto
	operator=(matrix_partition && other) noexcept
	-> matrix_partition &
	{
		this->~matrix_partition();
		new(this) matrix_partition(std::move(other));
		return *this;
	}

	auto
	operator()(size_t pos) noexcept
	-> T &
	{
		return base::operator()(pos);
	}

	auto
	operator()(size_t row, size_t col) noexcept
	-> T &
	{
		return base::operator()((row * m_cols) + col);
	}

	auto
	operator()(size_t row, size_t col) const noexcept
	-> T const &
	{
		return base::operator()((row * m_cols) + col);
	}

	auto
	handle_for_row(size_t row)
	-> starpu_data_handle_t
	{
		base::m_data_valid = false;
		return base::m_handles[row / m_part_rows];
	}

	auto
	flip_dim() noexcept
	-> void
	{
		std::swap(m_rows, m_cols);
	}

	auto
	getParent() noexcept
	-> matrix_partition &
	{
		return *this;
	}

	auto
	rows() const noexcept
	-> size_t
	{
		return m_rows;
	}

	auto
	cols() const noexcept
	-> size_t
	{
		return m_cols;
	}

	auto
	set(size_t const row, size_t const col, T const & val)
	-> void
	{
		base::set((row * m_cols) + col, val);
	}

	template<typename Iterator>
	auto
	set(Iterator begin, Iterator end)
	-> void
	{
		base::set(begin,end);
	}

	auto
	transpose_to(matrix_partition & other) noexcept
	-> void
	{
		if(!base::m_data_valid)
			base::allgather();
		transpose_local_store_to(other);
	}

	using base::allgather;
	using base::block_count_from;
	using base::capacity;
	using base::data;
	using base::fill;
	using base::gather_to_root;
	using base::local_storage_handle;
	using base::handle_for;
	using base::make_ext_w;
	using base::partition;
	using base::randomize;
	using base::scatter_from_root;
	using base::size;

private:
	auto
	alloc_local_storage() noexcept
	-> void override
	{
		if(!base::m_size)
			return;

		if(!base::m_data)
			base::m_data = new T[base::m_size];

		if(!base::m_external && !base::m_data_handle)
		{
			starpu_matrix_data_register(
				&(base::m_data_handle),
				STARPU_MAIN_RAM,
				(uintptr_t)(base::m_data),
				m_cols,
				m_cols,
				m_rows,
				sizeof(T));
			starpu_mpi_data_register(
				base::m_data_handle,
				skepu::cluster::mpi_tag(),
				STARPU_MPI_PER_NODE);
		}
	}

	auto
	alloc_partitions() noexcept
	-> void override
	{
		if(!base::m_size)
			return;

		if(!base::m_part_data)
			base::m_part_data = new T[base::m_part_size];

		for(size_t i(0); i < base::m_handles.size(); ++i)
		{
			auto & handle = base::m_handles[i];
			if(handle)
				continue;

			if(i == skepu::cluster::mpi_rank())
				starpu_register(base::m_part_data, m_part_rows, m_cols, handle, i);
			else
				starpu_register(0, m_part_rows, m_cols, handle, i);
		}
	}

	auto
	get_ptr(starpu_data_handle_t & handle) noexcept
	-> T * override
	{
		return (T *)starpu_matrix_get_local_ptr(handle);
	}

	auto
	update_sizes() noexcept
	-> void override
	{
		size_t size_arr[]{m_rows, m_cols};
		starpu_data_handle_t size_handle;
		starpu_variable_data_register(
			&size_handle,
			STARPU_MAIN_RAM,
			(uintptr_t)size_arr,
			sizeof(size_arr));
		starpu_mpi_data_register(size_handle, cluster::mpi_tag(), 0);
		starpu_mpi_get_data_on_all_nodes_detached(MPI_COMM_WORLD, size_handle);
		starpu_data_acquire(size_handle, STARPU_R);

		m_rows = size_arr[0];
		m_cols = size_arr[1];

		starpu_data_release(size_handle);
		starpu_data_unregister_no_coherency(size_handle);

		set_sizes();

		alloc_partitions();
		alloc_local_storage();
	}

	auto
	set_sizes()
	-> void
	{
		m_part_rows = m_rows / skepu::cluster::mpi_size();
		if(m_rows % skepu::cluster::mpi_size())
			++m_part_rows;
		size_t part_size = m_part_rows * m_cols;

		base::m_size = m_rows * m_cols;
		base::m_part_size = part_size;
		base::m_capacity = part_size * skepu::cluster::mpi_size();
	}

	auto
	starpu_register(
		T * ptr,
		size_t const rows,
		size_t const cols,
		starpu_data_handle_t & handle,
		int rank) noexcept
	-> void
	{
		if(ptr)
			starpu_matrix_data_register(
				&handle,
				STARPU_MAIN_RAM,
				(uintptr_t)ptr,
				cols,
				cols,
				rows,
				sizeof(T));
		else
			starpu_matrix_data_register(
				&handle,
				-1, // StarPU takes care of data management for other ranks handles.
				0,
				cols,
				cols,
				rows,
				sizeof(T));

		starpu_mpi_data_register(
			handle,
			cluster::mpi_tag(),
			rank);
	}

	auto
	transpose_local_store_to(matrix_partition & other) noexcept
	-> void
	{
		if(!other.m_data)
			other.alloc_local_storage();
		starpu_data_acquire(base::m_data_handle, STARPU_R);
		starpu_data_acquire(other.m_data_handle, STARPU_W);

		auto data_it = base::m_data;
		for(size_t i(0); i < m_rows; ++i)
			for(size_t j(0); j < m_cols; ++j, ++data_it)
				other(j,i) = *data_it;

		starpu_data_release(other.m_data_handle);
		starpu_data_release(base::m_data_handle);

		other.m_data_valid = true;
		other.m_part_valid = false;
	}
};

} // namespace util
} // namespace skepu

#endif // SKEPU_STARPU_MATRIX_PARTITION_HPP
