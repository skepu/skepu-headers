#pragma once
#ifndef SKEPU_STARPU_TENSOR4_HPP
#define SKEPU_STARPU_TENSOR4_HPP 1

#include "skepu3/cluster/common.hpp"

#include "../tensor3/iterator.hpp"
#include "proxy.hpp"
#include "partition.cpp"

namespace skepu
{

template<typename T>
class Tensor4
{
public:
	typedef size_t size_type;
	typedef T value_type;
	typedef T & reference;
	typedef T const & const_reference;
	typedef T * pointer;
	typedef ptrdiff_t difference_type;

	typedef Iterator<T, util::tensor4_partition> iterator;
	typedef Iterator<const T, util::tensor4_partition> const_iterator;

	typedef util::tensor4_partition<T> partition_type;

private:
	partition_type m_partition;

public:
	Tensor4() noexcept : m_partition() {}

	Tensor4(Tensor4 const & other) noexcept
	: m_partition(other.m_partition)
	{}

	Tensor4(Tensor4 && other) noexcept
	: m_partition(std::move(other))
	{}

	~Tensor4() noexcept = default;

	explicit
	Tensor4(size_type i, size_type j, size_type k, size_type l)
	: Tensor4(i, j, k, l, T())
	{}

	explicit
	Tensor4(
		size_type i,
		size_type j,
		size_type k,
		size_type l,
		const_reference val)
	: m_partition(i,j,k,l)
	{
		if(m_partition.size())
			m_partition.fill(m_partition.size(), val);
	}

	auto
	init(size_type i, size_type j, size_type k, size_type l) noexcept
	-> void
	{
		auto size = i * j * k * l;
		if(m_partition.size())
		{
			if(!cluster::mpi_rank())
				std::cerr << "[SkePU][Tensor4] Error: "
					"Can only be initialized once!\n";
			std::abort();
		}
		if(!(size))
		{
			if(!cluster::mpi_rank())
				std::cerr << "[SkePU][Tensor4] Error: "
					"Can not initialize to size of zero\n";
			std::abort();
		}

		m_partition.init(i, j, k, l);
	}

	auto
	init(
		size_type i,
		size_type j,
		size_type k,
		size_type l,
		const_reference val) noexcept
	-> void
	{
		init(i, j, k, l);
		m_partition.fill(m_partition.size(), val);
	}

	auto
	operator=(Tensor4 const & other) noexcept
	-> Tensor4 &
	{
		m_partition = other.m_partition;
		return *this;
	}

	auto
	operator=(Tensor4 && other) noexcept
	-> Tensor4 &
	{
		m_partition = std::move(other.m_partition);
		return *this;
	}

	auto
	operator()(size_type const pos) noexcept
	-> reference
	{
		return m_partition(pos);
	}

	auto
	operator()(size_type const pos) const noexcept
	-> const_reference
	{
		return m_partition(pos);
	}

	auto
	operator()(size_type i, size_type j, size_type k, size_type l) noexcept
	-> reference
	{
		return m_partition(i, j, k, l);
	}

	auto
	operator()(size_type i, size_type j, size_type k, size_type l) const noexcept
	-> const_reference
	{
		return m_partition(i, j, k, l);
	}

	auto
	front() noexcept
	-> reference
	{
		return m_partition(0);
	}

	auto
	front() const noexcept
	-> const_reference
	{
		return m_partition(0);
	}

	auto
	back() noexcept
	-> reference
	{
		return m_partition(m_partition.size() -1);
	}

	auto
	back() const noexcept
	-> const_reference
	{
		return m_partition(m_partition.size() -1);
	}

	auto
	begin() noexcept
	-> iterator
	{
		return iterator(m_partition);
	}

	auto
	begin() const noexcept
	-> const_iterator
	{
		return const_iterator(m_partition);
	}

	auto
	end() noexcept
	-> iterator
	{
		return iterator(m_partition.data() + m_partition.size(), m_partition);
	}

	auto
	end() const noexcept
	-> const_iterator
	{
		return const_iterator(m_partition.data() + m_partition.size(), m_partition);
	}

	auto
	swap(Tensor4 & other) noexcept
	-> void
	{
		auto tmp = std::move(m_partition);
		m_partition = std::move(other.m_partition);
		other.m_partition = std::move(tmp);
	}

	/* Capacity */
	auto
	size() const noexcept
	-> size_type
	{
		return m_partition.size();
	}

	auto
	size_i() const noexcept
	-> size_type
	{
		return m_partition.size_i();
	}

	auto
	size_j() const noexcept
	-> size_type
	{
		return m_partition.size_j();
	}

	auto
	size_k() const noexcept
	-> size_type
	{
		return m_partition.size_k();
	}

	auto constexpr
	size_l() const noexcept
	-> size_type
	{
		return m_partition.size_l();
	}

	auto
	empty() const noexcept
	-> bool
	{
		return !m_partition.size();
	}

	auto
	resize(size_type i, size_type j, size_type k, size_type l) noexcept
	-> void
	{
		resize(i, j, k, l, T());
	}

	auto
	resize(
		size_type i,
		size_type j,
		size_type k,
		size_type l,
		const_reference val) noexcept
	-> void
	{
		auto tmp = partition_type(i, j, k, l);
		auto cpy_i = std::min(i, m_partition.size_i());
		auto cpy_j = std::min(j, m_partition.size_j());
		auto cpy_k = std::min(k, m_partition.size_k());
		auto cpy_l = std::min(l, m_partition.size_l());

		for(size_t ii(0); ii < cpy_i; ++ii)
		{
			for(size_t ij(0); ij < cpy_j; ++ij)
			{
				for(size_t ik(0); ik < cpy_k; ++ik)
				{
					for(size_t il(0); il < cpy_l; ++il)
						tmp(ii, ij, ik, il) = m_partition(ii, ij, ik, il);
					for(size_t il(cpy_l); il < l; ++il)
						tmp(ii, ij, ik, il) = val;
				}
				for(size_t ik(cpy_k); ik < k; ++ik)
					for(size_t il(0); il < l; ++il)
						tmp(ii, ij, ik, il) = val;
			}
			for(size_t ij(cpy_j); ij < j; ++ij)
				for(size_t ik(0); ik < k; ++ik)
					for(size_t il(0); il < l; ++il)
						tmp(ii, ij, ik, il) = val;
		}
		for(size_t ii(cpy_i); ii < i; ++ii)
			for(size_t ij(cpy_j); ij < j; ++ij)
				for(size_t ik(0); ik < k; ++ik)
					for(size_t il(0); il < l; ++il)
						tmp(ii, ij, ik, il) = val;

		m_partition = std::move(tmp);
	}

	/* Consistency */
	auto
	flush(FlushMode /* e */ = FlushMode::Default) noexcept
	-> void
	{
		m_partition.allgather();
	}

	/* Utility functions */
	auto
	randomize(int /* n */, int /* x */)
	-> void
	{
		throw std::logic_error{std::string(__PRETTY_FUNCTION__)
			+ " Not implemented yet!\n"};
	}

	auto
	save(const std::string& /* filename */, const std::string& /* delimiter */)
	-> void
	{
		throw std::logic_error{std::string(__PRETTY_FUNCTION__)
			+ " Not implemented yet!\n"};
	}

	auto
	load(const std::string& /* e */, size_type /* s */)
	-> void
	{
		throw std::logic_error{std::string(__PRETTY_FUNCTION__)
			+ " Not implemented yet!\n"};
	}

private:
	friend cont;

	auto
	getParent() noexcept
	-> partition_type &
	{
		return m_partition;
	}

	auto
	getParent() const noexcept
	-> partition_type const &
	{
		return m_partition;
	}
};

template<typename T>
struct is_skepu_tensor4<Tensor4<T>> : public std::true_type {};

} // namespace skepu

namespace std {

template<typename T>
auto
swap(skepu::Tensor4<T> & a, skepu::Tensor4<T> & b) noexcept
-> void
{
	return a.swap(b);
}

} // namespace std

#endif // SKEPU_STARPU_TENSOR4_HPP
