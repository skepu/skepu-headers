#pragma once

#include "skepu3/impl/common.hpp"

namespace skepu
{
	namespace impl
	{
		template<typename, typename, typename...>
		class MapOverlap1D;
		
		template<typename, typename, typename...>
		class MapOverlap2D;
		
		template<typename, typename, typename...>
		class MapOverlap3D;
		
		template<typename, typename, typename...>
		class MapOverlap4D;
	}
	
	template<typename Ret, typename T, typename... Args>
	impl::MapOverlap1D<Ret, T, Args...> MapOverlapWrapper(std::function<Ret(Region1D<T>, Args...)> mapo)
	{
		return impl::MapOverlap1D<Ret, T, Args...>(mapo);
	}
	
	// For function pointers
	template<typename Ret, typename T, typename... Args>
	impl::MapOverlap1D<Ret, T, Args...> MapOverlap(Ret(*mapo)(Region1D<T>, Args...))
	{
		return MapOverlapWrapper((std::function<Ret(Region1D<T>, Args...)>)mapo);
	}
	
	// For lambdas and functors
	template<typename T>
	auto MapOverlap(T mapo) -> decltype(MapOverlapWrapper(lambda_cast(mapo)))
	{
		return MapOverlapWrapper(lambda_cast(mapo));
	}
	
	
	template<typename Ret, typename T, typename... Args>
	impl::MapOverlap2D<Ret, T, Args...> MapOverlapWrapper(std::function<Ret(Region2D<T>, Args...)> mapo)
	{
		return impl::MapOverlap2D<Ret, T, Args...>(mapo);
	}
	
	// For function pointers
	template<typename Ret, typename T, typename... Args>
	impl::MapOverlap2D<Ret, T, Args...> MapOverlap(Ret(*mapo)(Region2D<T>, Args...))
	{
		return MapOverlapWrapper((std::function<Ret(Region2D<T>, Args...)>)mapo);
	}
	
	
	
	
	template<typename Ret, typename T, typename... Args>
	impl::MapOverlap3D<Ret, T, Args...> MapOverlapWrapper(std::function<Ret(Region3D<T>, Args...)> mapo)
	{
		return impl::MapOverlap3D<Ret, T, Args...>(mapo);
	}
	
	// For function pointers
	template<typename Ret, typename T, typename... Args>
	impl::MapOverlap3D<Ret, T, Args...> MapOverlap(Ret(*mapo)(Region3D<T>, Args...))
	{
		return MapOverlapWrapper((std::function<Ret(Region3D<T>, Args...)>)mapo);
	}
	
	
	template<typename Ret, typename T, typename... Args>
	impl::MapOverlap4D<Ret, T, Args...> MapOverlapWrapper(std::function<Ret(Region4D<T>, Args...)> mapo)
	{
		return impl::MapOverlap4D<Ret, T, Args...>(mapo);
	}
	
	// For function pointers
	template<typename Ret, typename T, typename... Args>
	impl::MapOverlap4D<Ret, T, Args...> MapOverlap(Ret(*mapo)(Region4D<T>, Args...))
	{
		return MapOverlapWrapper((std::function<Ret(Region4D<T>, Args...)>)mapo);
	}
	
	
	enum class Edge
	{
		Cyclic, Duplicate, Pad
	};
	
	enum class Overlap
	{
		RowWise, ColWise, RowColWise, ColRowWise
	};
	
	namespace impl
	{
		template<typename Ret, typename T, typename... Args>
		class MapOverlapBase: public SeqSkeletonBase
		{
		protected:
		//	using T = typename std::remove_const<typename std::remove_pointer<Arg1>::type>::type;
			
		public:
			
			void setOverlapMode(Overlap mode)
			{
				this->m_overlapPolicy = mode;
			}
			
			void setEdgeMode(Edge mode)
			{
				this->m_edge = mode;
			}
			
			void setPad(T pad)
			{
				this->m_pad = pad;
			}
			
		protected:
			Overlap m_overlapPolicy = skepu::Overlap::RowWise;
			Edge m_edge = Edge::Duplicate;
			T m_pad {};
		};
		
		
		template<typename Ret, typename T, typename... Args>
		class MapOverlap1D: public MapOverlapBase<Ret, T, Args...>
		{
		//	static_assert(std::is_pointer<Arg1>::value, "Parameter must be of pointer type");
			
			using MapFunc = std::function<Ret(Region1D<T>, Args...)>;
		//	using T = typename MapOverlapBase<Ret, Arg1, Args...>::T;
			
		public:
			
			void setOverlap(size_t o)
			{
				this->m_overlap = o;
			}
			
			size_t getOverlap() const
			{
				return this->m_overlap;
			}
			
			template<template<class> class Container, size_t... AI, size_t... CI, typename... CallArgs>
			Container<Ret> &helper(Container<Ret> &res, Container<T> &arg, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
			{
				const int overlap = (int)this->m_overlap;
				const size_t size = arg.size();
				T start[3*overlap], end[3*overlap];
				
				for (size_t i = 0; i < overlap; ++i)
				{
					switch (this->m_edge)
					{
					case Edge::Cyclic:
						start[i] = arg[size + i  - overlap];
						end[3*overlap-1 - i] = arg[overlap-i-1];
						break;
					case Edge::Duplicate:
						start[i] = arg[0];
						end[3*overlap-1 - i] = arg[size-1];
						break;
					case Edge::Pad:
						start[i] = this->m_pad;
						end[3*overlap-1 - i] = this->m_pad;
					}
				}
				
				for (size_t i = overlap, j = 0; i < 3*overlap; ++i, ++j)
					start[i] = arg[j];
				
				for (size_t i = 0, j = 0; i < 2*overlap; ++i, ++j)
					end[i] = arg[j + size - 2*overlap];
				
				for (size_t i = 0; i < overlap; ++i)
					res[i] = this->mapFunc({overlap, 1, &start[i + overlap]}, get<AI>(args...).hostProxy()..., get<CI>(args...)...);
					
				for (size_t i = overlap; i < size - overlap; ++i)
					res[i] = this->mapFunc({overlap, 1, &arg[i]}, get<AI>(args...).hostProxy()..., get<CI>(args...)...);
					
				for (size_t i = size - overlap; i < size; ++i)
					res[i] = this->mapFunc({overlap, 1, &end[i + 2 * overlap - size]}, get<AI>(args...).hostProxy()..., get<CI>(args...)...);
				
				return res;
			}
			
			
			template<template<class> class Container, typename... CallArgs>
			Container<Ret> &operator()(Container<Ret> &res, Container<T>& arg, CallArgs&&... args)
			{
				constexpr size_t anyCont = trait_count_first<is_skepu_container, CallArgs...>::value;
				typename make_pack_indices<anyCont, 0>::type any_indices;
				typename make_pack_indices<sizeof...(CallArgs), anyCont>::type const_indices;
				
				this->helper(res, arg, any_indices, const_indices, args...);
				return res;
			}
			
			
			template<size_t... AI, size_t... CI, typename... CallArgs>
			void apply_colwise(skepu::Matrix<Ret>& res, skepu::Matrix<T>& arg, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
			{
				const int overlap = (int)this->m_overlap;
				size_t size = arg.size();
				T start[3*overlap], end[3*overlap];
				
				size_t rowWidth = arg.total_cols();
				size_t colWidth = arg.total_rows();
				size_t stride = rowWidth;
				
				const Ret *inputBegin = arg.getAddress();
				const Ret *inputEnd = inputBegin + size;
				Ret *outputBegin = res.getAddress();
				
				for(size_t col = 0; col < arg.total_cols(); ++col)
				{
					inputEnd = inputBegin + (rowWidth * (colWidth-1));
					
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
						outputBegin[i*stride] = this->mapFunc({overlap, 1, &start[i + overlap]},
							get<AI>(args...).hostProxy()..., get<CI>(args...)...);
						
					for (size_t i = overlap; i < colWidth - overlap; ++i)
						outputBegin[i*stride] = this->mapFunc({overlap, stride, &inputBegin[i*stride]},
							get<AI>(args...).hostProxy()..., get<CI>(args...)...);
						
					for (size_t i = colWidth - overlap; i < colWidth; ++i)
						outputBegin[i*stride] = this->mapFunc({overlap, 1, &end[i + 2 * overlap - colWidth]},
							get<AI>(args...).hostProxy()..., get<CI>(args...)...);
					
					inputBegin += 1;
					outputBegin += 1;
				}
			}
			
			
			template<size_t... AI, size_t... CI, typename... CallArgs>
			void apply_rowwise(skepu::Matrix<Ret>& res, skepu::Matrix<T>& arg, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
			{
				int overlap = (int)this->m_overlap;
				size_t size = arg.size();
				T start[3*overlap], end[3*overlap];
				
				size_t rowWidth = arg.total_cols();
				size_t stride = 1;
				
				const Ret *inputBegin = arg.getAddress();
				const Ret *inputEnd = inputBegin + size;
				Ret *out = res.getAddress();
				
				for (size_t row = 0; row < arg.total_rows(); ++row)
				{
					inputEnd = inputBegin + rowWidth;
					
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
						out[i] = this->mapFunc({overlap, stride, &start[i + overlap]}, get<AI>(args...).hostProxy()..., get<CI>(args...)...);
						
					for (size_t i = overlap; i < rowWidth - overlap; ++i)
						out[i] = this->mapFunc({overlap, stride, &inputBegin[i]}, get<AI>(args...).hostProxy()..., get<CI>(args...)...);
						
					for (size_t i = rowWidth - overlap; i < rowWidth; ++i)
						out[i] = this->mapFunc({overlap, stride, &end[i + 2 * overlap - rowWidth]}, get<AI>(args...).hostProxy()..., get<CI>(args...)...);
					
					inputBegin += rowWidth;
					out += rowWidth;
				}
			}
			
			
			template<typename... CallArgs>
			Matrix<Ret> &operator()(Matrix<Ret> &res, Matrix<T>& arg, CallArgs&&... args)
			{
				if (arg.total_rows() != res.total_rows() || arg.total_cols() != res.total_cols())
					SKEPU_ERROR("Non-matching container sizes");
				
				constexpr size_t anyCont = trait_count_first<is_skepu_container, CallArgs...>::value;
				typename make_pack_indices<anyCont, 0>::type any_indices;
				typename make_pack_indices<sizeof...(CallArgs), anyCont>::type const_indices;
				
				switch (this->m_overlapPolicy)
				{
					case Overlap::RowColWise: {
						skepu::Matrix<Ret> tmp_m(res.total_rows(), res.total_cols());
						this->apply_rowwise(tmp_m, arg, any_indices, const_indices, args...);
						this->apply_colwise(res, tmp_m, any_indices, const_indices, args...);
						break;
					}
					case Overlap::ColRowWise: {
						skepu::Matrix<Ret> tmp_m(res.total_rows(), res.total_cols());
						this->apply_colwise(tmp_m, arg, any_indices, const_indices, args...);
						this->apply_rowwise(res, tmp_m, any_indices, const_indices, args...);
						break;
					}
					case Overlap::ColWise:
						this->apply_colwise(res, arg, any_indices, const_indices, args...);
						break;
						
					case Overlap::RowWise:
						this->apply_rowwise(res, arg, any_indices, const_indices, args...);
						break;
						
					default:
						SKEPU_ERROR("MapOverlap: Invalid overlap policy");
				}
				
				return res;
			}
			
		private:
			MapFunc mapFunc;
			MapOverlap1D(MapFunc map): mapFunc(map) {}
			
			size_t m_overlap = 0;
			
			friend MapOverlap1D<Ret, T, Args...> skepu::MapOverlapWrapper<Ret, T, Args...>(MapFunc);
		};
		
		
		template<typename Ret, typename T, typename... Args>
		class MapOverlap2D: public MapOverlapBase<Ret, T, Args...>
		{
		//	static_assert(std::is_pointer<Arg1>::value, "Parameter must be of pointer type");
			
			using MapFunc = std::function<Ret(Region2D<T>, Args...)>;
		//	using T = typename MapOverlapBase<Ret, Arg1, Args...>::T;
			
		public:
			void setBackend(BackendSpec) {}
			void resetBackend() {}
			
			void setOverlap(size_t o)
			{
				this->m_overlapX = o;
				this->m_overlapY = o;
			}
			
			void setOverlap(size_t y, size_t x)
			{
				this->m_overlapX = x;
				this->m_overlapY = y;
			}
			
			std::pair<size_t, size_t> getOverlap() const
			{
				return std::make_pair(this->m_overlap_x, this->m_overlap_y);
			}
			
			template<size_t... AI, size_t... CI, typename... CallArgs>
			void apply_helper(Matrix<Ret> &res, Matrix<T> &arg, pack_indices<AI...>, pack_indices<CI...>,  CallArgs&&... args)
			{
				const int overlap_x = (int)this->m_overlapX;
				const int overlap_y = (int)this->m_overlapY;
				const size_t in_rows = arg.total_rows();
				const size_t in_cols = arg.total_cols();
				const size_t out_rows = res.total_rows();
				const size_t out_cols = res.total_cols();
				
				if ((in_rows - overlap_y*2 != out_rows) && (in_cols - overlap_x*2 != out_cols))
					SKEPU_ERROR("Non-matching container sizes");
				
				for (size_t i = 0; i < out_rows; i++)
					for (size_t j = 0; j < out_cols; j++)
						res(i, j) = this->mapFunc({overlap_x, overlap_y, in_cols, &arg((i+overlap_y)*in_cols + (j+overlap_x))},
							get<AI>(args...).hostProxy()..., get<CI>(args...)...);
			}
			
			template<typename... CallArgs>
			Matrix<Ret> &operator()(Matrix<Ret> &res, Matrix<T>& arg, CallArgs&&... args)
			{
				constexpr size_t anyCont = trait_count_first<is_skepu_container, CallArgs...>::value;
				typename make_pack_indices<anyCont, 0>::type any_indices;
				typename make_pack_indices<sizeof...(CallArgs), anyCont>::type const_indices;
				apply_helper(res, arg, any_indices, const_indices, args...);
				return res;
			}
			
		private:
			MapFunc mapFunc;
			MapOverlap2D(MapFunc map): mapFunc(map) {}
			
			size_t m_overlapX, m_overlapY;
			
			friend MapOverlap2D<Ret, T, Args...> skepu::MapOverlapWrapper<Ret, T, Args...>(MapFunc);
		};
		
		
		
		
		
		
		
		template<typename Ret, typename T, typename... Args>
		class MapOverlap3D: public MapOverlapBase<Ret, T, Args...>
		{
			using MapFunc = std::function<Ret(Region3D<T>, Args...)>;
			
		public:
			void setBackend(BackendSpec) {}
			void resetBackend() {}
			
			void setOverlap(int o)
			{
				this->m_overlap_i = o;
				this->m_overlap_j = o;
				this->m_overlap_k = o;
			}
			
			void setOverlap(int oi, int oj, int ok)
			{
				this->m_overlap_i = oi;
				this->m_overlap_j = oj;
				this->m_overlap_k = ok;
			}
			
			std::tuple<int, int, int> getOverlap() const
			{
				return std::make_tuple(this->m_overlap_i, this->m_overlap_j, this->m_overlap_k);
			}
			
			template<size_t... AI, size_t... CI, typename... CallArgs>
			void apply_helper(Tensor3<Ret> &res, Tensor3<T> &arg, pack_indices<AI...>, pack_indices<CI...>,  CallArgs&&... args)
			{
				if (  (arg.size_i() - this->m_overlap_i*2 != res.size_i())
					&& (arg.size_j() - this->m_overlap_j*2 != res.size_j())
					&& (arg.size_k() - this->m_overlap_k*2 != res.size_k()))
					SKEPU_ERROR("Non-matching container sizes");
				
				for (size_t i = 0; i < res.size_i(); i++)
					for (size_t j = 0; j < res.size_j(); j++)
						for (size_t k = 0; k < res.size_k(); k++)
							res(i, j, k) = this->mapFunc(Region3D<T>{this->m_overlap_i, this->m_overlap_j, this->m_overlap_k,
								arg.size_i(), arg.size_j(), &arg(i+this->m_overlap_i, + j+this->m_overlap_j, + k+this->m_overlap_k)},
								get<AI>(args...).hostProxy()..., get<CI>(args...)...);
			}
			
			template<typename... CallArgs>
			Tensor3<Ret> &operator()(Tensor3<Ret> &res, Tensor3<T>& arg, CallArgs&&... args)
			{
				constexpr size_t anyCont = trait_count_first<is_skepu_container, CallArgs...>::value;
				typename make_pack_indices<anyCont, 0>::type any_indices;
				typename make_pack_indices<sizeof...(CallArgs), anyCont>::type const_indices;
				apply_helper(res, arg, any_indices, const_indices, args...);
				return res;
			}
			
		private:
			MapFunc mapFunc;
			MapOverlap3D(MapFunc map): mapFunc(map) {}
			
			int m_overlap_i, m_overlap_j, m_overlap_k;
			
			friend MapOverlap3D<Ret, T, Args...> skepu::MapOverlapWrapper<Ret, T, Args...>(MapFunc);
		};
		
		
		
		
		
		template<typename Ret, typename T, typename... Args>
		class MapOverlap4D: public MapOverlapBase<Ret, T, Args...>
		{
			using MapFunc = std::function<Ret(Region4D<T>, Args...)>;
			
		public:
			void setBackend(BackendSpec) {}
			void resetBackend() {}
			
			void setOverlap(int o)
			{
				this->m_overlap_i = o;
				this->m_overlap_j = o;
				this->m_overlap_k = o;
				this->m_overlap_l = o;
			}
			
			void setOverlap(int oi, int oj, int ok, int ol)
			{
				this->m_overlap_i = oi;
				this->m_overlap_j = oj;
				this->m_overlap_k = ok;
				this->m_overlap_l = ol;
			}
			
			std::tuple<int, int, int, int> getOverlap() const
			{
				return std::make_tuple(this->m_overlap_i, this->m_overlap_j, this->m_overlap_k, this->m_overlap_l);
			}
			
			template<size_t... AI, size_t... CI, typename... CallArgs>
			void apply_helper(Tensor4<Ret> &res, Tensor4<T> &arg, pack_indices<AI...>, pack_indices<CI...>,  CallArgs&&... args)
			{
				if (  (arg.size_i() - this->m_overlap_i*2 != res.size_i())
					&& (arg.size_j() - this->m_overlap_j*2 != res.size_j())
					&& (arg.size_k() - this->m_overlap_k*2 != res.size_k())
					&& (arg.size_l() - this->m_overlap_l*2 != res.size_l()))
					SKEPU_ERROR("Non-matching container sizes");
				
				for (size_t i = 0; i < res.size_i(); i++)
					for (size_t j = 0; j < res.size_j(); j++)
						for (size_t k = 0; k < res.size_k(); k++)
							for (size_t l = 0; l < res.size_l(); l++)
							{
								res(i, j, k, l) = this->mapFunc(Region4D<T>{this->m_overlap_i, this->m_overlap_j, this->m_overlap_k, this->m_overlap_l,
									arg.size_i(), arg.size_j(), arg.size_k(), &arg(i + this->m_overlap_i, j + this->m_overlap_j, k + this->m_overlap_k, l + this->m_overlap_l)},
									get<AI>(args...).hostProxy()..., get<CI>(args...)...);
							}
			}
			
			template<typename... CallArgs>
			Tensor4<Ret> &operator()(Tensor4<Ret> &res, Tensor4<T>& arg, CallArgs&&... args)
			{
				constexpr size_t anyCont = trait_count_first<is_skepu_container, CallArgs...>::value;
				typename make_pack_indices<anyCont, 0>::type any_indices;
				typename make_pack_indices<sizeof...(CallArgs), anyCont>::type const_indices;
				apply_helper(res, arg, any_indices, const_indices, args...);
				return res;
			}
			
		private:
			MapFunc mapFunc;
			MapOverlap4D(MapFunc map): mapFunc(map) {}
			
			int m_overlap_i, m_overlap_j, m_overlap_k, m_overlap_l;
			
			friend MapOverlap4D<Ret, T, Args...> skepu::MapOverlapWrapper<Ret, T, Args...>(MapFunc);
		};
		
	}
}
