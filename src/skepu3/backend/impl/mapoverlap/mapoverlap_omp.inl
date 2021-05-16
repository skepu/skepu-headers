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
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel>
		::vector_OpenMP(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			auto &arg = get<outArity>(args...);
			// Sync with device data
			arg.updateHost();
			pack_expand((get<AI, CallArgs...>(args...).getParent().updateHost(hasReadAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			pack_expand((get<AI, CallArgs...>(args...).getParent().invalidateDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			pack_expand((get<OI, CallArgs...>(args...).getParent().invalidateDeviceData(), 0)...);
			
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
			{
				auto res = F::forward(MapOverlapFunc::OMP, Index1D{i}, Region1D<T>{overlap, stride, &start[i + overlap]},
						get<AI, CallArgs...>(args...).hostProxy()..., get<CI, CallArgs...>(args...)...);
				SKEPU_VARIADIC_RETURN(get<OI>(args...)(i)..., res);
			}
				
#pragma omp parallel for schedule(runtime)
			for (size_t i = overlap; i < size - overlap; ++i)
			{
				auto res = F::forward(MapOverlapFunc::OMP, Index1D{i}, Region1D<T>{overlap, stride, &arg(i)},
					get<AI, CallArgs...>(args...).hostProxy()..., get<CI, CallArgs...>(args...)...);
				SKEPU_VARIADIC_RETURN(get<OI>(args...)(i)..., res);
			}
				
			for (size_t i = size - overlap; i < size; ++i)
			{
				auto res = F::forward(MapOverlapFunc::OMP, Index1D{i}, Region1D<T>{overlap, stride, &end[i + 2 * overlap - size]},
					get<AI, CallArgs...>(args...).hostProxy()..., get<CI, CallArgs...>(args...)...);
				SKEPU_VARIADIC_RETURN(get<OI>(args...)(i)..., res);
			}
		}
		
		
		/*!
		 *  Performs the row-wise MapOverlap on a range of elements on the \em OpenMP with a seperate output range.
		 *  Used internally by other methods to apply row-wise mapoverlap operation.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel>
		::rowwise_OpenMP(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			auto &arg = get<outArity>(args...);
			// Sync with device data
			arg.updateHost();
			pack_expand((get<AI, CallArgs...>(args...).getParent().updateHost(hasReadAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			pack_expand((get<AI, CallArgs...>(args...).getParent().invalidateDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			pack_expand((get<OI, CallArgs...>(args...).getParent().invalidateDeviceData(), 0)...);
			
			int overlap = (int)this->m_overlap;
			size_t size = arg.size();
			T start[3*overlap], end[3*overlap];
			
			size_t rowWidth = arg.total_cols();
			size_t stride = 1;
			
			const T *inputBegin = arg.getAddress();
			const T *inputEnd = inputBegin + size;
			
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
				{
					auto res = F::forward(MapOverlapFunc::OMP, Index1D{i}, Region1D<T>{overlap, stride, &start[i + overlap]}, get<AI, CallArgs...>(args...).hostProxy()..., get<CI, CallArgs...>(args...)...);
					SKEPU_VARIADIC_RETURN(get<OI>(args...)(row, i)..., res);
				}
					
#pragma omp parallel for schedule(runtime)
				for (size_t i = overlap; i < rowWidth - overlap; ++i)
				{
					auto res = F::forward(MapOverlapFunc::OMP, Index1D{i}, Region1D<T>{overlap, stride, &inputBegin[i]}, get<AI, CallArgs...>(args...).hostProxy()..., get<CI, CallArgs...>(args...)...);
					SKEPU_VARIADIC_RETURN(get<OI>(args...)(row, i)..., res);
				}
					
				for (size_t i = rowWidth - overlap; i < rowWidth; ++i)
				{
					auto res = F::forward(MapOverlapFunc::OMP, Index1D{i}, Region1D<T>{overlap, stride, &end[i + 2 * overlap - rowWidth]}, get<AI, CallArgs...>(args...).hostProxy()..., get<CI, CallArgs...>(args...)...);
					SKEPU_VARIADIC_RETURN(get<OI>(args...)(row, i)..., res);
				}
				
				inputBegin += rowWidth;
			}
		}
		
		
		/*!
		 *  Performs the column-wise MapOverlap on a range of elements on the \em OpenMP with a seperate output range.
		 *  Used internally by other methods to apply column-wise mapoverlap operation.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel>
		::colwise_OpenMP(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			auto &arg = get<outArity>(args...);
			// Sync with device data
			arg.updateHost();
			pack_expand((get<AI, CallArgs...>(args...).getParent().updateHost(hasReadAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			pack_expand((get<AI, CallArgs...>(args...).getParent().invalidateDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			pack_expand((get<OI, CallArgs...>(args...).getParent().invalidateDeviceData(), 0)...);
			
			int overlap = (int)this->m_overlap;
			size_t size = arg.size();
			T start[3*overlap], end[3*overlap];
			
			size_t rowWidth = arg.total_cols();
			size_t colWidth = arg.total_rows();
			size_t stride = rowWidth;
			
			const T *inputBegin = arg.getAddress();
			const T *inputEnd = inputBegin + size;
			
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
				{
					auto res = F::forward(MapOverlapFunc::OMP, Index1D{i}, Region1D<T>{overlap, 1, &start[i + overlap]},
						get<AI, CallArgs...>(args...).hostProxy()..., get<CI, CallArgs...>(args...)...);
					SKEPU_VARIADIC_RETURN(get<OI>(args...)(i, col)..., res);
				}
					
#pragma omp parallel for schedule(runtime)
				for (size_t i = overlap; i < colWidth - overlap; ++i)
				{
					auto res = F::forward(MapOverlapFunc::OMP, Index1D{i}, Region1D<T>{overlap, stride, &inputBegin[i*stride]},
						get<AI, CallArgs...>(args...).hostProxy()..., get<CI, CallArgs...>(args...)...);
					SKEPU_VARIADIC_RETURN(get<OI>(args...)(i, col)..., res);
				}
					
				for (size_t i = colWidth - overlap; i < colWidth; ++i)
				{
					auto res = F::forward(MapOverlapFunc::OMP, Index1D{i}, Region1D<T>{overlap, 1, &end[i + 2 * overlap - colWidth]},
						get<AI, CallArgs...>(args...).hostProxy()..., get<CI, CallArgs...>(args...)...);
					SKEPU_VARIADIC_RETURN(get<OI>(args...)(i, col)..., res);
				}
				
				inputBegin += 1;
			}
		}
		
		
		template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap2D<MapOverlapFunc, CUDAKernel, CLKernel>
		::helper_OpenMP(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			size_t size_i = get<0>(args...).size_i();
			size_t size_j = get<0>(args...).size_j();
			
			// Sync with device data
			pack_expand((get<EI, CallArgs...>(args...).getParent().updateHost(), 0)...);
			pack_expand((get<AI, CallArgs...>(args...).getParent().updateHost(hasReadAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			pack_expand((get<AI, CallArgs...>(args...).getParent().invalidateDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			pack_expand((get<OI, CallArgs...>(args...).getParent().invalidateDeviceData(), 0)...);
			
			auto &arg = get<outArity>(args...);
			
			Region2D<T> region{arg, this->m_overlap_y, this->m_overlap_x, this->m_edge, this->m_pad};
			
			Index2D start{0, 0}, end{size_i, size_j};
			if (this->m_edge == Edge::None)
			{
				start = Index2D{(size_t)this->m_overlap_y, (size_t)this->m_overlap_x};
				end = Index2D{size_i - this->m_overlap_y, size_j - this->m_overlap_x};
			}
			
#pragma omp parallel for schedule(runtime)
			for (size_t i = start.col; i < end.col; i++)
				for (size_t j = start.row; j < end.row; j++)
					if (p == Parity::None || index_parity(p, i, j))
					{
						region.idx = Index2D{i,j};
						auto res = F::forward(MapOverlapFunc::CPU, Index2D{i,j}, region, get<AI>(args...).hostProxy()..., get<CI>(args...)...);
						SKEPU_VARIADIC_RETURN(get<OI>(args...)(i, j)..., res);
					}
		}
		
		template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI,  typename... CallArgs>
		void MapOverlap3D<MapOverlapFunc, CUDAKernel, CLKernel>
		::helper_OpenMP(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			size_t size_i = get<0>(args...).size_i();
			size_t size_j = get<0>(args...).size_j();
			size_t size_k = get<0>(args...).size_k();
			
			// Sync with device data
			pack_expand((get<EI, CallArgs...>(args...).getParent().updateHost(), 0)...);
			pack_expand((get<AI, CallArgs...>(args...).getParent().updateHost(hasReadAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			pack_expand((get<AI, CallArgs...>(args...).getParent().invalidateDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			pack_expand((get<OI, CallArgs...>(args...).getParent().invalidateDeviceData(), 0)...);
			
			auto &arg = get<outArity>(args...);
			
			Region3D<T> region{arg, this->m_overlap_i, this->m_overlap_j, this->m_overlap_k, this->m_edge, this->m_pad};
			
			Index3D start{0, 0, 0}, end{size_i, size_j, size_k};
			if (this->m_edge == Edge::None)
			{
				start = Index3D{(size_t)this->m_overlap_i, (size_t)this->m_overlap_j, (size_t)this->m_overlap_k};
				end = Index3D{size_i - this->m_overlap_i, size_j - this->m_overlap_j, size_k - this->m_overlap_k};
			}
			
#pragma omp parallel for schedule(runtime)
			for (size_t i = start.i; i < end.i; i++)
				for (size_t j = start.j; j < end.j; j++)
					for (size_t k = start.k; k < end.k; k++)
						if (p == Parity::None || index_parity(p, i, j, k))
						{
							region.idx = Index3D{i,j,k};
							auto res = F::forward(MapOverlapFunc::CPU, Index3D{i,j,k}, region, get<AI>(args...).hostProxy()..., get<CI>(args...)...);
							SKEPU_VARIADIC_RETURN(get<OI>(args...)(i, j, k)..., res);
						}
		}
		
		
		
		
		template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs, REQUIRES_B(!MapOverlapFunc::usesPRNG && true && sizeof...(CallArgs) > 0)>
		void MapOverlap4D<MapOverlapFunc, CUDAKernel, CLKernel>
		::helper_OpenMP(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			size_t size_i = get<0>(args...).size_i();
			size_t size_j = get<0>(args...).size_j();
			size_t size_k = get<0>(args...).size_k();
			size_t size_l = get<0>(args...).size_l();
			
			// Sync with device data
			pack_expand((get<EI, CallArgs...>(args...).getParent().updateHost(), 0)...);
			pack_expand((get<AI, CallArgs...>(args...).getParent().updateHost(hasReadAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			pack_expand((get<AI, CallArgs...>(args...).getParent().invalidateDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			pack_expand((get<OI, CallArgs...>(args...).getParent().invalidateDeviceData(), 0)...);
			
			auto &arg = get<outArity>(args...);
			
			Region4D<T> region{arg, this->m_overlap_i, this->m_overlap_j, this->m_overlap_k, this->m_overlap_l, this->m_edge, this->m_pad};
			
			Index4D start{0, 0, 0, 0}, end{size_i, size_j, size_k, size_l};
			if (this->m_edge == Edge::None)
			{
				start = Index4D{(size_t)this->m_overlap_i, (size_t)this->m_overlap_j, (size_t)this->m_overlap_k, (size_t)this->m_overlap_l};
				end = Index4D{size_i - this->m_overlap_i, size_j - this->m_overlap_j, size_k - this->m_overlap_k, size_l - this->m_overlap_l};
			}
			
#pragma omp parallel for schedule(runtime)
			for (size_t i = start.i; i < end.i; i++)
				for (size_t j = start.j; j < end.j; j++)
					for (size_t k = start.k; k < end.k; k++)
						for (size_t l = start.l; l < end.l; l++)
							if (p == Parity::None || index_parity(p, i, j, k, l))
							{
								region.idx = Index4D{i,j,k,l};
								auto res = F::forward(MapOverlapFunc::CPU, Index4D{i,j,k,l}, region, get<AI>(args...).hostProxy()..., get<CI>(args...)...);
								SKEPU_VARIADIC_RETURN(get<OI>(args...)(i, j, k, l)..., res);
							}
		}
		
		// PRNG
		template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs, REQUIRES_B(!!MapOverlapFunc::usesPRNG && sizeof...(CallArgs) > 0)>
		void MapOverlap4D<MapOverlapFunc, CUDAKernel, CLKernel>
		::helper_OpenMP(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			size_t size_i = get<0>(args...).size_i();
			size_t size_j = get<0>(args...).size_j();
			size_t size_k = get<0>(args...).size_k();
			size_t size_l = get<0>(args...).size_l();
			
			// Sync with device data
			pack_expand((get<EI, CallArgs...>(args...).getParent().updateHost(), 0)...);
			pack_expand((get<AI, CallArgs...>(args...).getParent().updateHost(hasReadAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			pack_expand((get<AI, CallArgs...>(args...).getParent().invalidateDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			pack_expand((get<OI, CallArgs...>(args...).getParent().invalidateDeviceData(), 0)...);
			
			auto &arg = get<outArity>(args...);
			
			Region4D<T> region{arg, this->m_overlap_i, this->m_overlap_j, this->m_overlap_k, this->m_overlap_l, this->m_edge, this->m_pad};
			
			Index4D start{0, 0, 0, 0}, end{size_i, size_j, size_k, size_l};
			if (this->m_edge == Edge::None)
			{
				start = Index4D{(size_t)this->m_overlap_i, (size_t)this->m_overlap_j, (size_t)this->m_overlap_k, (size_t)this->m_overlap_l};
				end = Index4D{size_i - this->m_overlap_i, size_j - this->m_overlap_j, size_k - this->m_overlap_k, size_l - this->m_overlap_l};
			}
			
			if (this->m_prng == nullptr)
				SKEPU_ERROR("No random stream set in skeleton instance");

			size_t threads = std::min<size_t>((end.i - start.i), omp_get_max_threads());
			size_t final_size = (end.i - start.i) * (end.j - start.j) * (end.k - start.k) * (end.l - start.l);
			size_t work_unit = (end.j - start.j) * (end.k - start.k) * (end.l - start.l);
			auto randoms = this->m_prng->template asRandom<MapOverlapFunc::randomCount>(final_size, threads, work_unit);

#ifdef SKEPU_PRNG_VERIFY_FORWARD
			for (auto &r : randoms)
			{
				std::cout << "State: " << r.m_state << " valid uses: " << r.m_valid_uses << "\n";
			}
#endif
			
#pragma omp parallel for schedule(static)
			for (size_t i = start.i; i < end.i; i++)
				for (size_t j = start.j; j < end.j; j++)
					for (size_t k = start.k; k < end.k; k++)
						for (size_t l = start.l; l < end.l; l++)
							if (p == Parity::None || index_parity(p, i, j, k, l))
							{
								auto &my_random = randoms(omp_get_thread_num());
								region.idx = Index4D{i,j,k,l};
								auto res = F::forward(MapOverlapFunc::CPU, Index4D{i,j,k,l}, my_random, region, get<AI>(args...).hostProxy()..., get<CI>(args...)...);
								SKEPU_VARIADIC_RETURN(get<OI>(args...)(i, j, k, l)..., res);
							}
							
#ifdef SKEPU_PRNG_VERIFY_FORWARD
			for (auto &r : randoms)
			{
				std::cout << "State: " << r.m_state << " valid uses: " << r.m_valid_uses << "\n";
			}
#endif
		}
		
	} // namespace backend
} // namespace skepu

#endif
