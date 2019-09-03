#ifndef VECTOR_ITERATOR_HPP
#define VECTOR_ITERATOR_HPP

#include <skepu2/cluster/matrix_iterator.hpp>

namespace skepu2
{
	template<typename T>
	using VectorIterator = MatrixIterator<T>;
}

#endif /* VECTOR_ITERATOR_HPP */
