/*! \file vector.h
*  \brief Contains a class declaration for the Vector container.
 */

#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cstddef>
#include <map>

#include "backend/malloc_allocator.h"

#ifdef SKEPU_PRECOMPILED

#include "backend/environment.h"
#include "backend/device_mem_pointer_cl.h"
#include "backend/device_mem_pointer_cu.h"

#endif // SKEPU_PRECOMPILED


namespace skepu
{
	// TENSOR 3
	
	template<typename T>
	class Tensor3;
	
	// Proxy vector for user functions
	template<typename T>
	struct Ten3
	{
		using ContainerType = Tensor3<T>;
		
		Ten3(T *dataptr, size_t si, size_t sj, size_t sk):
			data{dataptr}, size_i{si}, size_j{sj}, size_k{sk} {}
		Ten3(): data{nullptr}, size_i{0}, size_j{0}, size_k{0} {} // empty proxy constructor
		
		T &operator[](size_t index)       { return this->data[index]; }
		T  operator[](size_t index) const { return this->data[index]; }
		
		T &operator()(size_t i, size_t j, size_t k)       { return this->data[i * this->size_j * this->size_k + j * this->size_k + k]; }
		T  operator()(size_t i, size_t j, size_t k) const { return this->data[i * this->size_j * this->size_k + j * this->size_k + k]; }
		
		T *data;
		size_t size_i, size_j, size_k;
	};
	
	
	template <typename T>
	class Tensor3Iterator;
	
	template<typename T>
	class Tensor3 : public Vector<T>
	{
	public:
		using size_type = typename Vector<T>::size_type;
		using proxy_type = Ten3<T>;
		
#ifdef SKEPU_CUDA
		using device_pointer_type_cu = typename Vector<T>::device_pointer_type_cu;
#endif
		
		typedef Tensor3Iterator<T> iterator;
		typedef Tensor3Iterator<const T> const_iterator;
		
		friend class Tensor3Iterator<T>;
		
		explicit Tensor3()
		: m_size_i(0), m_size_j(0), m_size_k(0), Vector<T>{}
		{}
			
		explicit Tensor3(size_type si, size_type sj, size_type sk, const T& val = T())
		: m_size_i(si), m_size_j(sj), m_size_k(sk), Vector<T>(si * sj * sk, val)
		{}
	
		void init(size_type si, size_type sj, size_type sk)
		{
			Vector<T>::init(si * sj * sk);
			this->m_size_i = si;
			this->m_size_j = sj;
			this->m_size_k = sk;
		}
		
		void init(size_type si, size_type sj, size_type sk, const T& val)
		{
			Vector<T>::init(si * sj * sk, val);
			this->m_size_i = si;
			this->m_size_j = sj;
			this->m_size_k = sk;
		}
		
		iterator begin()
		{
			return iterator(*this, &Vector<T>::m_data[0]);
		}
		
		iterator end()
		{
			return iterator(*this, &Vector<T>::m_data[this->m_size]);
		}
		
		template<typename Ignore>
		proxy_type hostProxy(ProxyTag::Default, Ignore)
		{
			return { this->m_data, this->m_size_i, this->m_size_j, this->m_size_k };
		}
		
		proxy_type hostProxy() { return this->hostProxy(ProxyTag::Default{}, 0); }
		
		size_type size() const
		{
			return Vector<T>::m_size;
		}
		
		size_type size_i() const
		{
			return m_size_i;
		}
		
		size_type size_j() const
		{
			return m_size_j;
		}
		
		size_type size_k() const
		{
			return m_size_k;
		}
		
		size_type size_l() const
		{
			return 0;
		}
		
		// All dimensions
		std::tuple<size_type, size_type, size_type> size_info() const
		{
			return {this->m_size_i, this->m_size_j, this->m_size_k};
		}
		
		T& operator()(size_type i, size_type j, size_type k)
		{
			return Vector<T>::m_data[i * this->m_size_j * this->m_size_k + j * this->m_size_k + k];
		}
		
		const T& operator()(size_type i, size_type j, size_type k) const
		{
			return Vector<T>::m_data[i * this->m_size_j * this->m_size_k + j * this->m_size_k + k];
		}
		
		const Tensor3<T>& getParent() const { return *this; }
		Tensor3<T>& getParent() { return *this; }
		
#ifdef SKEPU_CUDA
		template<typename Ignore>
		std::pair<device_pointer_type_cu, proxy_type>
		cudaProxy(size_t deviceID, AccessMode accessMode, ProxyTag::Default, Ignore)
		{
			device_pointer_type_cu devptr = this->updateDevice_CU(this->m_data, this->size(), deviceID, accessMode);
			proxy_type proxy;
			proxy.data = devptr->getDeviceDataPointer();
			proxy.size_i = this->m_size_i;
			proxy.size_j = this->m_size_j;
			proxy.size_k = this->m_size_k;
			return {devptr, proxy};
		}
		
		std::pair<device_pointer_type_cu, proxy_type>
		cudaProxy(size_t deviceID, AccessMode accessMode)
		{
			return this->cudaProxy(deviceID, accessMode, ProxyTag::Default{}, 0);
		}
#endif // SKEPU_CUDA
		
	private:
		
		size_type m_size_i, m_size_j, m_size_k;
		
	};
	
	
	template <typename T>
	class Tensor3Iterator : public std::iterator<std::random_access_iterator_tag, T>
	{
	public:
		typedef Tensor3Iterator<T> iterator;
		typedef Tensor3Iterator<const T> const_iterator;
	   typedef typename std::conditional<std::is_const<T>::value,
						const Tensor3<typename std::remove_const<T>::type>, Tensor3<T>>::type parent_type;
		
		using proxy_type = typename parent_type::proxy_type;
	
	public: //-- Constructors & Destructor --//
		
		Tensor3Iterator(parent_type& vec, T *std_iterator);
		
	public: //-- Types --//
		
#ifdef SKEPU_CUDA
		typedef typename parent_type::device_pointer_type_cu device_pointer_type_cu;
#endif
		
#ifdef SKEPU_OPENCL
		typedef typename parent_type::device_pointer_type_cl device_pointer_type_cl;
#endif
		
	public: //-- Extras --//
		
		Index3D getIndex() const;
		
		parent_type& getParent() const;
		iterator& begin(); // returns itself
		size_t size(); // returns number of elements "left" in parent container from this index
		
		T* getAddress() const;
		
		template<typename Ignore>
		proxy_type hostProxy(ProxyTag::Default, Ignore)
		{ return {this->m_std_iterator, this->size()}; }
		
		proxy_type hostProxy() { return this->hostProxy(ProxyTag::Default{}, 0); }
		
		// Does not care about device data, use with care...sometimes pass negative indices...
		T& operator()(const ssize_t index = 0);
		const T& operator()(const ssize_t index) const;
		
	public: //-- Operators --//
		
		T& operator[](const ssize_t index);
		const T& operator[](const ssize_t index) const;
		
		bool operator==(const iterator& i);
		bool operator!=(const iterator& i);
		bool operator<(const iterator& i);
		bool operator>(const iterator& i);
		bool operator<=(const iterator& i);
		bool operator>=(const iterator& i);
		
		const iterator& operator++();
		iterator operator++(int);
		const iterator& operator--();
		iterator operator--(int);
		
		const iterator& operator+=(const ssize_t i);
		const iterator& operator-=(const ssize_t i);
		
		iterator operator-(const ssize_t i) const;
		iterator operator+(const ssize_t i) const;
		
		typename parent_type::difference_type operator-(const iterator& i) const;
		
		T& operator *();
		const T& operator* () const;
		
		const T& operator-> () const;
		T& operator-> ();
		
	protected: //-- Data --//
		
		parent_type& m_parent;
		T *m_std_iterator;
	};
	
	
	
	
	
	
	// TENSOR 4
	
	template<typename T>
	class Tensor4;
	
	// Proxy vector for user functions
	template<typename T>
	struct Ten4
	{
		using ContainerType = Tensor4<T>;
		
		Ten4(T *dataptr, size_t si, size_t sj, size_t sk, size_t sl):
			data{dataptr}, size_i{si}, size_j{sj}, size_k{sk}, size_l{sl} {}
		Ten4(): data{nullptr}, size_i{0}, size_j{0}, size_k{0}, size_l{0} {} // empty proxy constructor
		
		T &operator[](size_t index)       { return this->data[index]; }
		T  operator[](size_t index) const { return this->data[index]; }
		
		T &operator()(size_t i, size_t j, size_t k, size_t l)
			{ return this->data[i * this->size_j * this->size_k * this->size_l + j * this->size_k * this->size_l + k * this->size_l + l]; }
		T  operator()(size_t i, size_t j, size_t k, size_t l) const
			{ return this->data[i * this->size_j * this->size_k * this->size_l + j * this->size_k * this->size_l + k * this->size_l + l]; }
		
		T *data;
		size_t size_i, size_j, size_k, size_l;
	};
	
	
	template <typename T>
	class Tensor4Iterator;
	
	template<typename T>
	class Tensor4 : public Vector<T>
	{
	public:
		using size_type = typename Vector<T>::size_type;
		using proxy_type = Ten4<T>;
		
#ifdef SKEPU_CUDA
		using device_pointer_type_cu = typename Vector<T>::device_pointer_type_cu;
#endif
		
		typedef Tensor4Iterator<T> iterator;
		typedef Tensor4Iterator<const T> const_iterator;
		
		friend class Tensor4Iterator<T>;
		
		explicit Tensor4()
		: m_size_i(0), m_size_j(0), m_size_k(0), m_size_l(0), Vector<T>()
		{}
		
		explicit Tensor4(size_type si, size_type sj, size_type sk, size_type sl, const T& val = T())
		: m_size_i(si), m_size_j(sj), m_size_k(sk), m_size_l(sl),
		Vector<T>(si * sj * sk * sl, val)
		{}
		
		void init(size_type si, size_type sj, size_type sk, size_type sl)
		{
			Vector<T>::init(si * sj * sk * sl);
			this->m_size_i = si;
			this->m_size_j = sj;
			this->m_size_k = sk;
			this->m_size_l = sl;
		}
		
		void init(size_type si, size_type sj, size_type sk, size_type sl, const T& val)
		{
			Vector<T>::init(si * sj * sk * sl, val);
			this->m_size_i = si;
			this->m_size_j = sj;
			this->m_size_k = sk;
			this->m_size_l = sl;
		}
		
		iterator begin()
		{
			return iterator(*this, &Vector<T>::m_data[0]);
		}
		
		iterator end()
		{
			return iterator(*this, &Vector<T>::m_data[this->m_size]);
		}
		
		template<typename Ignore>
		proxy_type hostProxy(ProxyTag::Default, Ignore)
		{
			return { this->m_data, this->m_size_i, this->m_size_j, this->m_size_k, this->m_size_l };
		}
		
		proxy_type hostProxy() { return this->hostProxy(ProxyTag::Default{}, 0); }
		
		size_type size() const
		{
			return Vector<T>::m_size;
		}
		
		size_type size_i() const
		{
			return m_size_i;
		}
		
		size_type size_j() const
		{
			return m_size_j;
		}
		
		size_type size_k() const
		{
			return m_size_k;
		}
		
		size_type size_l() const
		{
			return m_size_l;
		}
		
		// All dimensions
		std::tuple<size_type, size_type, size_type, size_type> size_info() const
		{
			return {this->m_size_i, this->m_size_j, this->m_size_k, this->m_size_l};
		}
		
		T& operator()(size_type i, size_type j, size_type k, size_type l)
		{
			return Vector<T>::m_data[i * this->m_size_j * this->m_size_k * this->m_size_l + j * this->m_size_k * this->m_size_l + k * this->m_size_l + l];
		}
		
		const T& operator()(size_type i, size_type j, size_type k, size_type l) const
		{
			return Vector<T>::m_data[i * this->m_size_j * this->m_size_k * this->m_size_l + j * this->m_size_k * this->m_size_l + k * this->m_size_l + l];
		}
		
		const Tensor4<T>& getParent() const { return *this; }
		Tensor4<T>& getParent() { return *this; }
		
#ifdef SKEPU_CUDA
		template<typename Ignore>
		std::pair<device_pointer_type_cu, proxy_type>
		cudaProxy(size_t deviceID, AccessMode accessMode, ProxyTag::Default, Ignore)
		{
			device_pointer_type_cu devptr = this->updateDevice_CU(this->m_data, this->size(), deviceID, accessMode);
			proxy_type proxy;
			proxy.data = devptr->getDeviceDataPointer();
			proxy.size_i = this->m_size_i;
			proxy.size_j = this->m_size_j;
			proxy.size_k = this->m_size_k;
			proxy.size_l = this->m_size_l;
			return {devptr, proxy};
		}
		
		std::pair<device_pointer_type_cu, proxy_type>
		cudaProxy(size_t deviceID, AccessMode accessMode)
		{
			return this->cudaProxy(deviceID, accessMode, ProxyTag::Default{}, 0);
		}
#endif // SKEPU_CUDA
		
	private:
		
		size_type m_size_i, m_size_j, m_size_k, m_size_l;
		
	};
	
	
	
	template <typename T>
	class Tensor4Iterator : public std::iterator<std::random_access_iterator_tag, T>
	{
	public:
		typedef Tensor4Iterator<T> iterator;
		typedef Tensor4Iterator<const T> const_iterator;
	   typedef typename std::conditional<std::is_const<T>::value,
						const Tensor4<typename std::remove_const<T>::type>, Tensor4<T>>::type parent_type;
		
		using proxy_type = typename parent_type::proxy_type;
	
	public: //-- Constructors & Destructor --//
		
		Tensor4Iterator(parent_type& vec, T *std_iterator);
		
	public: //-- Types --//
		
#ifdef SKEPU_CUDA
		typedef typename parent_type::device_pointer_type_cu device_pointer_type_cu;
#endif
		
#ifdef SKEPU_OPENCL
		typedef typename parent_type::device_pointer_type_cl device_pointer_type_cl;
#endif
		
	public: //-- Extras --//
		
		Index4D getIndex() const;
		
		parent_type& getParent() const;
		iterator& begin(); // returns itself
		size_t size(); // returns number of elements "left" in parent container from this index
		
		T* getAddress() const;
		
		template<typename Ignore>
		proxy_type hostProxy(ProxyTag::Default, Ignore)
		{ return {this->m_std_iterator, this->size()}; }
		
		proxy_type hostProxy() { return this->hostProxy(ProxyTag::Default{}, 0); }
		
		// Does not care about device data, use with care...sometimes pass negative indices...
		T& operator()(const ssize_t index = 0);
		const T& operator()(const ssize_t index) const;
		
	public: //-- Operators --//
		
		T& operator[](const ssize_t index);
		const T& operator[](const ssize_t index) const;
		
		bool operator==(const iterator& i);
		bool operator!=(const iterator& i);
		bool operator<(const iterator& i);
		bool operator>(const iterator& i);
		bool operator<=(const iterator& i);
		bool operator>=(const iterator& i);
		
		const iterator& operator++();
		iterator operator++(int);
		const iterator& operator--();
		iterator operator--(int);
		
		const iterator& operator+=(const ssize_t i);
		const iterator& operator-=(const ssize_t i);
		
		iterator operator-(const ssize_t i) const;
		iterator operator+(const ssize_t i) const;
		
		typename parent_type::difference_type operator-(const iterator& i) const;
		
		T& operator *();
		const T& operator* () const;
		
		const T& operator-> () const;
		T& operator-> ();
		
	protected: //-- Data --//
		
		parent_type& m_parent;
		T *m_std_iterator;
	};
}

#include "backend/impl/tensor/tensor_iterator.inl"
