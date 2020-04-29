/*! \file mapoverlap_omp.inl
*  \brief Contains the definitions of OpenMP specific member functions for the MapOverlap skeleton.
 */

#ifdef SKEPU_OPENMP

#include <omp.h>

namespace skepu
{
	namespace backend
	{
		/*!
		 *  Performs the MapOverlap on a range of elements using \em OpenMP as backend and a seperate output range.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel>
		template<template<class> class Container, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel>
		::vector_OpenMP(Container<Ret>& res, Container<T>& arg, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			// Sync with device data
			arg.updateHost();
			pack_expand((get<AI, CallArgs...>(args...).getParent().updateHost(hasReadAccess(MapOverlapFunc::anyAccessMode[AI])), 0)...);
			pack_expand((get<AI, CallArgs...>(args...).getParent().invalidateDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI])), 0)...);
			res.invalidateDeviceData();
			
			const int overlap = (int)this->m_overlap;
			const size_t size = arg.size();
			const size_t stride = 1;
			
			T start[3*overlap], end[3*overlap];
			
			omp_set_num_threads(this->m_selected_spec->CPUThreads());
			
#pragma omp parallel for schedule(runtime)
			for (size_t i = 0; i < overlap; ++i)
			{
				switch (this->m_edge)
				{
				case Edge::Cyclic:
					start[i] = arg(size + i  - overlap);
					end[3*overlap-1 - i] = arg(overlap-i-1);
					break;
				case Edge::Duplicate:
					start[i] = arg(0);
					end[3*overlap-1 - i] = arg(size-1);
					break;
				case Edge::Pad:
					start[i] = this->m_pad;
					end[3*overlap-1 - i] = this->m_pad;
				}
			}
			
			for (size_t i = overlap, j = 0; i < 3*overlap; ++i, ++j)
				start[i] = arg(j);
			
			for (size_t i = 0, j = 0; i < 2*overlap; ++i, ++j)
				end[i] = arg(j + size - 2*overlap);
			
			for (size_t i = 0; i < overlap; ++i)
				res(i) = MapOverlapFunc::OMP({overlap, stride, &start[i + overlap]},
						get<AI, CallArgs...>(args...).hostProxy()..., get<CI, CallArgs...>(args...)...);
				
#pragma omp parallel for schedule(runtime)
			for (size_t i = overlap; i < size - overlap; ++i)
				res(i) = MapOverlapFunc::OMP({overlap, stride, &arg(i)},
					get<AI, CallArgs...>(args...).hostProxy()..., get<CI, CallArgs...>(args...)...);
				
			for (size_t i = size - overlap; i < size; ++i)
				res(i) = MapOverlapFunc::OMP({overlap, stride, &end[i + 2 * overlap - size]},
					get<AI, CallArgs...>(args...).hostProxy()..., get<CI, CallArgs...>(args...)...);
		}
		
		
		/*!
		 *  Performs the row-wise MapOverlap on a range of elements on the \em OpenMP with a seperate output range.
		 *  Used internally by other methods to apply row-wise mapoverlap operation.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel>
		template<size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel>
		::rowwise_OpenMP(skepu::Matrix<Ret>& res, skepu::Matrix<T>& arg, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			// Sync with device data
			arg.updateHost();
			pack_expand((get<AI, CallArgs...>(args...).getParent().updateHost(hasReadAccess(MapOverlapFunc::anyAccessMode[AI])), 0)...);
			pack_expand((get<AI, CallArgs...>(args...).getParent().invalidateDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI])), 0)...);
			res.invalidateDeviceData();
			
			int overlap = (int)this->m_overlap;
			size_t size = arg.size();
			T start[3*overlap], end[3*overlap];
			
			size_t rowWidth = arg.total_cols();
			size_t stride = 1;
			
			const Ret *inputBegin = arg.getAddress();
			const Ret *inputEnd = inputBegin + size;
			Ret *out = res.getAddress();
			
			omp_set_num_threads(this->m_selected_spec->CPUThreads());
			
			for (size_t row = 0; row < arg.total_rows(); ++row)
			{
				inputEnd = inputBegin + rowWidth;
				
#pragma omp parallel for schedule(runtime)
				for (size_t i = 0; i < overlap; ++i)
				{
					switch (this->m_edge)
					{
					case Edge::Cyclic:
						start[i] = inputEnd[i  - overlap];
						end[3*overlap-1 - i] = inputBegin[overlap-i-1];
						break;
					case Edge::Duplicate:
						start[i] = inputBegin[0];
						end[3*overlap-1 - i] = inputEnd[-1];
						break;
					case Edge::Pad:
						start[i] = this->m_pad;
						end[3*overlap-1 - i] = this->m_pad;
						break;
					}
				}
				
				for (size_t i = overlap, j = 0; i < 3*overlap; ++i, ++j)
					start[i] = inputBegin[j];
				
				for (size_t i = 0, j = 0; i < 2*overlap; ++i, ++j)
					end[i] = inputEnd[j - 2*overlap];
				
				for (size_t i = 0; i < overlap; ++i)
					out[i] = MapOverlapFunc::OMP({overlap, stride, &start[i + overlap]}, get<AI, CallArgs...>(args...).hostProxy()..., get<CI, CallArgs...>(args...)...);
					
#pragma omp parallel for schedule(runtime)
				for (size_t i = overlap; i < rowWidth - overlap; ++i)
					out[i] = MapOverlapFunc::OMP({overlap, stride, &inputBegin[i]}, get<AI, CallArgs...>(args...).hostProxy()..., get<CI, CallArgs...>(args...)...);
					
				for (size_t i = rowWidth - overlap; i < rowWidth; ++i)
					out[i] = MapOverlapFunc::OMP({overlap, stride, &end[i + 2 * overlap - rowWidth]}, get<AI, CallArgs...>(args...).hostProxy()..., get<CI, CallArgs...>(args...)...);
				
				inputBegin += rowWidth;
				out += rowWidth;
			}
		}
		
		
		/*!
		 *  Performs the column-wise MapOverlap on a range of elements on the \em OpenMP with a seperate output range.
		 *  Used internally by other methods to apply column-wise mapoverlap operation.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel>
		template<size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel>
		::colwise_OpenMP(skepu::Matrix<Ret>& res, skepu::Matrix<T>& arg, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			// Sync with device data
			arg.updateHost();
			pack_expand((get<AI, CallArgs...>(args...).getParent().updateHost(hasReadAccess(MapOverlapFunc::anyAccessMode[AI])), 0)...);
			pack_expand((get<AI, CallArgs...>(args...).getParent().invalidateDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI])), 0)...);
			res.invalidateDeviceData();
			
			int overlap = (int)this->m_overlap;
			size_t size = arg.size();
			T start[3*overlap], end[3*overlap];
			
			size_t rowWidth = arg.total_cols();
			size_t colWidth = arg.total_rows();
			size_t stride = rowWidth;
			
			const Ret *inputBegin = arg.getAddress();
			const Ret *inputEnd = inputBegin + size;
			
			omp_set_num_threads(this->m_selected_spec->CPUThreads());
			
			for (size_t col = 0; col < arg.total_cols(); ++col)
			{
				inputEnd = inputBegin + (rowWidth * (colWidth-1));
				
#pragma omp parallel for schedule(runtime)
				for (size_t i = 0; i < overlap; ++i)
				{
					switch (this->m_edge)
					{
					case Edge::Cyclic:
						start[i] = inputEnd[(i+1-overlap)*stride];
						end[3*overlap-1 - i] = inputBegin[(overlap-i-1)*stride];
						break;
					case Edge::Duplicate:
						start[i] = inputBegin[0];
						end[3*overlap-1 - i] = inputEnd[0]; // hmmm...
						break;
					case Edge::Pad:
						start[i] = this->m_pad;
						end[3*overlap-1 - i] = this->m_pad;
						break;
					}
				}
				
				for (size_t i = overlap, j = 0; i < 3*overlap; ++i, ++j)
					start[i] = inputBegin[j*stride];
				
				for (size_t i = 0, j = 0; i < 2*overlap; ++i, ++j)
					end[i] = inputEnd[(j - 2*overlap + 1)*stride];
				
				for (size_t i = 0; i < overlap; ++i)
					res(i * stride + col) = MapOverlapFunc::OMP({overlap, 1, &start[i + overlap]},
						get<AI, CallArgs...>(args...).hostProxy()..., get<CI, CallArgs...>(args...)...);
					
#pragma omp parallel for schedule(runtime)
				for (size_t i = overlap; i < colWidth - overlap; ++i)
					res(i * stride + col) = MapOverlapFunc::OMP({overlap, stride, &inputBegin[i*stride]},
						get<AI, CallArgs...>(args...).hostProxy()..., get<CI, CallArgs...>(args...)...);
					
				for (size_t i = colWidth - overlap; i < colWidth; ++i)
					res(i * stride + col) = MapOverlapFunc::OMP({overlap, 1, &end[i + 2 * overlap - colWidth]},
						get<AI, CallArgs...>(args...).hostProxy()..., get<CI, CallArgs...>(args...)...);
				
				inputBegin += 1;
			}
		}
		
		
		template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap2D<MapOverlapFunc, CUDAKernel, CLKernel>
		::helper_OpenMP(skepu::Matrix<Ret>& res, skepu::Matrix<T>& arg, pack_indices<AI...>, pack_indices<CI...>,  CallArgs&&... args)
		{
			// Sync with device data
			arg.updateHost();
			pack_expand((get<AI, CallArgs...>(args...).getParent().updateHost(hasReadAccess(MapOverlapFunc::anyAccessMode[AI])), 0)...);
			pack_expand((get<AI, CallArgs...>(args...).getParent().invalidateDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI])), 0)...);
			res.invalidateDeviceData();
			
			omp_set_num_threads(this->m_selected_spec->CPUThreads());
			
			const int overlap_x = (int)this->m_overlap_x;
			const int overlap_y = (int)this->m_overlap_y;
			const size_t rows = res.total_rows();
			const size_t cols = res.total_cols();
			const size_t in_cols = arg.total_cols();
			
#pragma omp parallel for schedule(runtime)
			for (size_t i = 0; i < rows; i++)
				for (size_t j = 0; j < cols; j++)
					res(i, j) = MapOverlapFunc::CPU({overlap_x, overlap_y, in_cols, &arg((i + overlap_y) * in_cols + (j + overlap_x))}, get<AI, CallArgs...>(args...).hostProxy()..., get<CI, CallArgs...>(args...)...);
		}
		
		template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap3D<MapOverlapFunc, CUDAKernel, CLKernel>
		::helper_OpenMP(skepu::Tensor3<Ret>& res, skepu::Tensor3<T>& arg, pack_indices<AI...>, pack_indices<CI...>,  CallArgs&&... args)
		{
			// Sync with device data
			arg.updateHost();
			pack_expand((get<AI, CallArgs...>(args...).getParent().updateHost(hasReadAccess(MapOverlapFunc::anyAccessMode[AI])), 0)...);
			pack_expand((get<AI, CallArgs...>(args...).getParent().invalidateDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI])), 0)...);
			res.invalidateDeviceData();
			
#pragma omp parallel for schedule(runtime)
			for (size_t i = 0; i < res.size_i(); i++)
				for (size_t j = 0; j < res.size_j(); j++)
					for (size_t k = 0; k < res.size_k(); k++)
						res(i, j, k) = MapOverlapFunc::CPU(Region3D<T>{this->m_overlap_i, this->m_overlap_j, this->m_overlap_k,
							arg.size_i(), arg.size_j(), &arg(i+this->m_overlap_i, + j+this->m_overlap_j, + k+this->m_overlap_k)},
							get<AI, CallArgs...>(args...).hostProxy()..., get<CI, CallArgs...>(args...)...);
		}
		
		
		
		
		template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap4D<MapOverlapFunc, CUDAKernel, CLKernel>
		::helper_OpenMP(skepu::Tensor4<Ret>& res, skepu::Tensor4<T>& arg, pack_indices<AI...>, pack_indices<CI...>,  CallArgs&&... args)
		{
			// Sync with device data
			arg.updateHost();
			pack_expand((get<AI, CallArgs...>(args...).getParent().updateHost(hasReadAccess(MapOverlapFunc::anyAccessMode[AI])), 0)...);
			pack_expand((get<AI, CallArgs...>(args...).getParent().invalidateDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI])), 0)...);
			res.invalidateDeviceData();
			
#pragma omp parallel for schedule(runtime)
			for (size_t i = 0; i < res.size_i(); i++)
				for (size_t j = 0; j < res.size_j(); j++)
					for (size_t k = 0; k < res.size_k(); k++)
						for (size_t l = 0; l < res.size_l(); l++)
						{
							res(i, j, k, l) = MapOverlapFunc::CPU(Region4D<T>{this->m_overlap_i, this->m_overlap_j, this->m_overlap_k, this->m_overlap_l,
								arg.size_i(), arg.size_j(), arg.size_k(), &arg(i + this->m_overlap_i, j + this->m_overlap_j, k + this->m_overlap_k, l + this->m_overlap_l)},
								get<AI, CallArgs...>(args...).hostProxy()..., get<CI, CallArgs...>(args...)...);
						}
		}
		
	} // namespace backend
} // namespace skepu

#endif
