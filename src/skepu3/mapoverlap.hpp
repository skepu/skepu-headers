#pragma once

#include "skepu3/impl/common.hpp"
#include "skepu3/impl/region.hpp"

namespace skepu
{
	namespace impl
	{
		template<typename, typename...>
		class MapOverlap1D;
		
		template<typename, typename...>
		class MapOverlap2D;
		
		template<typename, typename...>
		class MapOverlap3D;
		
		template<typename, typename...>
		class MapOverlap4D;
	}
	
	template<typename Ret, typename... Args, REQUIRES(mapoverlap_dimensionality<Args...>::value == 1)>
	impl::MapOverlap1D<Ret, Args...> MapOverlapWrapper(std::function<Ret(Args...)> mapo)
	{
		return impl::MapOverlap1D<Ret, Args...>(mapo);
	}
	/*
	// For function pointers
	template<typename Ret, typename... Args>
	impl::MapOverlap1D<Ret, Args...> MapOverlap(Ret(*mapo)(Region1D<T>, Args...))
	{
		return MapOverlapWrapper((std::function<Ret(Region1D<T>, Args...)>)mapo);
	}*/
	
	
	template<typename Ret, typename... Args, REQUIRES(mapoverlap_dimensionality<Args...>::value == 2)>
	impl::MapOverlap2D<Ret, Args...> MapOverlapWrapper(std::function<Ret(Args...)> mapo)
	{
		return impl::MapOverlap2D<Ret, Args...>(mapo);
	}
	
	template<typename Ret, typename... Args, REQUIRES(mapoverlap_dimensionality<Args...>::value == 3)>
	impl::MapOverlap3D<Ret, Args...> MapOverlapWrapper(std::function<Ret(Args...)> mapo)
	{
		return impl::MapOverlap3D<Ret, Args...>(mapo);
	}
	
	template<typename Ret, typename... Args, REQUIRES(mapoverlap_dimensionality<Args...>::value == 4)>
	impl::MapOverlap4D<Ret, Args...> MapOverlapWrapper(std::function<Ret(Args...)> mapo)
	{
		return impl::MapOverlap4D<Ret, Args...>(mapo);
	}
	
	// For function pointers
	template<typename Ret, typename... Args>
	auto MapOverlap(Ret(*mapo)(Args...)) -> decltype(MapOverlapWrapper((std::function<Ret(Args...)>)mapo))
	{
		return MapOverlapWrapper((std::function<Ret(Args...)>)mapo);
	}
	
	// For lambdas and functors
	template<typename T>
	auto MapOverlap(T mapo) -> decltype(MapOverlapWrapper(lambda_cast(mapo)))
	{
		return MapOverlapWrapper(lambda_cast(mapo));
	}
	
	
	
	
	namespace impl
	{
		template<typename T>
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
		
		
		template<typename Ret, typename... Args>
		class MapOverlap1D: public MapOverlapBase<typename region_type<typename pack_element<is_indexed<Args...>::value ? 1 : 0, Args...>::type>::type>
		{
			static constexpr bool indexed = is_indexed<Args...>::value;
			static constexpr size_t InArity = 1;
			static constexpr size_t OutArity = out_size<Ret>::value;
			static constexpr size_t numArgs = sizeof...(Args) - (indexed ? 1 : 0) + OutArity;
			static constexpr size_t anyCont = trait_count_all<is_skepu_container_proxy, Args...>::value;
			
			static constexpr typename make_pack_indices<OutArity, 0>::type out_indices{};
			static constexpr typename make_pack_indices<InArity + OutArity, OutArity>::type elwise_indices{};
			static constexpr typename make_pack_indices<InArity + OutArity + anyCont, InArity + OutArity>::type any_indices{};
			static constexpr typename make_pack_indices<numArgs, InArity + OutArity + anyCont>::type const_indices{};
			
			using MapFunc = std::function<Ret(Args...)>;
			using F = ConditionalIndexForwarder<indexed, MapFunc>;
			using RegionType = typename pack_element<indexed ? 1 : 0, Args...>::type;
			using T = typename region_type<RegionType>::type;
			
		public:
			
			void setOverlap(size_t o)
			{
				this->m_overlap = o;
			}
			
			size_t getOverlap() const
			{
				return this->m_overlap;
			}
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void apply(pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
			{
				auto &arg = get<OutArity>(args...);
				
				const int overlap = (int)this->m_overlap;
				const size_t size = arg.size();
				
				// Verify overlap radius is valid
				if (this->m_edge != Edge::None && size < this->m_overlap * 2)
					SKEPU_ERROR("Non-matching overlap radius");
				
				if (disjunction(get<OI>(args...).size() < size...))
					SKEPU_ERROR("Non-matching output container sizes");
					
				if (this->m_edge != Edge::None && disjunction(get<EI>(args...).size() != size...))
					SKEPU_ERROR("Non-matching input container sizes");
				
				if (this->m_edge == Edge::None && disjunction(get<EI>(args...).size() - this->m_overlap*2 != size...))
					SKEPU_ERROR("Non-matching input container sizes");
				
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
						break;
					default:
						SKEPU_ERROR("Edge mode not supported");
					}
				}
				
				for (size_t i = overlap, j = 0; i < 3*overlap; ++i, ++j)
					start[i] = arg[j];
				
				for (size_t i = 0, j = 0; i < 2*overlap; ++i, ++j)
					end[i] = arg[j + size - 2*overlap];
				
				for (size_t i = 0; i < overlap; ++i)
				{
					auto res = F::forward(this->mapFunc, Index1D{i}, Region1D<T>{overlap, 1, &start[i + overlap]}, get<AI>(args...).hostProxy()..., get<CI>(args...)...);
					SKEPU_VARIADIC_RETURN(get<OI>(args...)(i)..., res);
				}
					
				for (size_t i = overlap; i < size - overlap; ++i)
				{
					auto res = F::forward(this->mapFunc, Index1D{i}, Region1D<T>{overlap, 1, &arg[i]}, get<AI>(args...).hostProxy()..., get<CI>(args...)...);
					SKEPU_VARIADIC_RETURN(get<OI>(args...)(i)..., res);
				}
					
				for (size_t i = size - overlap; i < size; ++i)
				{
					auto res = F::forward(this->mapFunc, Index1D{i}, Region1D<T>{overlap, 1, &end[i + 2 * overlap - size]}, get<AI>(args...).hostProxy()..., get<CI>(args...)...);
					SKEPU_VARIADIC_RETURN(get<OI>(args...)(i)..., res);
				}
			}
			
			
			template<typename... CallArgs,
				REQUIRES(is_skepu_vector<typename std::remove_reference<typename pack_element<0, CallArgs...>::type>::type>::value)>
			auto operator()(CallArgs&&... args) -> decltype(get<0>(args...))
			{
				this->apply(out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
				return get<0>(args...);
			}
			
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void apply_colwise(pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
			{
				size_t size_i = get<0>(args...).size_i();
				size_t size_j = get<0>(args...).size_j();
				
				// Verify overlap radius is valid
				if (this->m_edge != Edge::None && size_i < this->m_overlap * 2)
					SKEPU_ERROR("Non-matching overlap radius");
				
				if (disjunction(
					(get<OI>(args...).size_i() < size_i) &&
					(get<OI>(args...).size_j() < size_j) ...))
					SKEPU_ERROR("Non-matching output container sizes");
					
				if (this->m_edge != Edge::None && disjunction(
					(get<EI>(args...).size_i() != size_i) &&
					(get<EI>(args...).size_j() != size_j) ...))
					SKEPU_ERROR("Non-matching input container sizes");
				
				if (this->m_edge == Edge::None && disjunction(
					(get<EI>(args...).size_i() - this->m_overlap*2 != size_i) &&
					(get<EI>(args...).size_j() - this->m_overlap*2 != size_j) ...))
					SKEPU_ERROR("Non-matching input container sizes");
				
				auto &arg = get<OutArity>(args...);
				
				const int overlap = (int)this->m_overlap;
				size_t size = arg.size();
				T start[3*overlap], end[3*overlap];
				
				size_t rowWidth = arg.total_cols();
				size_t colWidth = arg.total_rows();
				size_t stride = rowWidth;
				
				const T *inputBegin = arg.getAddress();
				const T *inputEnd = inputBegin + size;
				
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
						default:
							SKEPU_ERROR("Edge mode not supported");
						}
					}
					
					for (size_t i = overlap, j = 0; i < 3*overlap; ++i, ++j)
						start[i] = inputBegin[j*stride];
					
					for (size_t i = 0, j = 0; i < 2*overlap; ++i, ++j)
						end[i] = inputEnd[(j - 2*overlap + 1)*stride];
					
					for (size_t i = 0; i < overlap; ++i)
					{
						auto res = F::forward(this->mapFunc, Index1D{i}, Region1D<T>{overlap, 1, &start[i + overlap]},
							get<AI>(args...).hostProxy()..., get<CI>(args...)...);
						SKEPU_VARIADIC_RETURN(get<OI>(args...)(i, col)..., res);
					}
						
					for (size_t i = overlap; i < colWidth - overlap; ++i)
					{
						auto res = F::forward(this->mapFunc, Index1D{i}, Region1D<T>{overlap, stride, &inputBegin[i*stride]},
							get<AI>(args...).hostProxy()..., get<CI>(args...)...);
						SKEPU_VARIADIC_RETURN(get<OI>(args...)(i, col)..., res);
					}
						
					for (size_t i = colWidth - overlap; i < colWidth; ++i)
					{
						auto res = F::forward(this->mapFunc, Index1D{i}, Region1D<T>{overlap, 1, &end[i + 2 * overlap - colWidth]},
							get<AI>(args...).hostProxy()..., get<CI>(args...)...);
						SKEPU_VARIADIC_RETURN(get<OI>(args...)(i, col)..., res);
					}
					
					inputBegin += 1;
				}
			}
			
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void apply_rowwise(pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
			{
				size_t size_i = get<0>(args...).size_i();
				size_t size_j = get<0>(args...).size_j();
				
				// Verify overlap radius is valid
				if (this->m_edge != Edge::None && size_j < this->m_overlap * 2)
					SKEPU_ERROR("Non-matching overlap radius");
				
				if (disjunction(
					(get<OI>(args...).size_i() < size_i) &&
					(get<OI>(args...).size_j() < size_j) ...))
					SKEPU_ERROR("Non-matching output container sizes");
					
				if (this->m_edge != Edge::None && disjunction(
					(get<EI>(args...).size_i() != size_i) &&
					(get<EI>(args...).size_j() != size_j) ...))
					SKEPU_ERROR("Non-matching input container sizes");
				
				if (this->m_edge == Edge::None && disjunction(
					(get<EI>(args...).size_i() - this->m_overlap*2 != size_i) &&
					(get<EI>(args...).size_j() - this->m_overlap*2 != size_j) ...))
					SKEPU_ERROR("Non-matching input container sizes");
				
				auto &arg = get<OutArity>(args...);
				
				int overlap = (int)this->m_overlap;
				size_t size = arg.size();
				T start[3*overlap], end[3*overlap];
				
				size_t rowWidth = arg.total_cols();
				size_t stride = 1;
				
				const T *inputBegin = arg.getAddress();
				const T *inputEnd = inputBegin + size;
				
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
						default:
							SKEPU_ERROR("Edge mode not supported");
						}
					}
					
					
					for (size_t i = overlap, j = 0; i < 3*overlap; ++i, ++j)
						start[i] = inputBegin[j];
					
					for (size_t i = 0, j = 0; i < 2*overlap; ++i, ++j)
						end[i] = inputEnd[j - 2*overlap];
					
					for (size_t i = 0; i < overlap; ++i)
					{
						auto res = F::forward(this->mapFunc, Index1D{i}, Region1D<T>{overlap, stride, &start[i + overlap]}, get<AI>(args...).hostProxy()..., get<CI>(args...)...);
						SKEPU_VARIADIC_RETURN(get<OI>(args...)(row, i)..., res);
					}
						
					for (size_t i = overlap; i < rowWidth - overlap; ++i)
					{
						auto res = F::forward(this->mapFunc, Index1D{i}, Region1D<T>{overlap, stride, &inputBegin[i]}, get<AI>(args...).hostProxy()..., get<CI>(args...)...);
						SKEPU_VARIADIC_RETURN(get<OI>(args...)(row, i)..., res);
					}
						
					for (size_t i = rowWidth - overlap; i < rowWidth; ++i)
					{
					 	auto res = F::forward(this->mapFunc, Index1D{i}, Region1D<T>{overlap, stride, &end[i + 2 * overlap - rowWidth]}, get<AI>(args...).hostProxy()..., get<CI>(args...)...);
						SKEPU_VARIADIC_RETURN(get<OI>(args...)(row, i)..., res);
					}
					
					inputBegin += rowWidth;
				}
			}
			
			
			template<typename... CallArgs, 
				REQUIRES(is_skepu_matrix<typename std::remove_reference<typename pack_element<0, CallArgs...>::type>::type>::value)>
			auto operator()(CallArgs&&... args) -> decltype(get<0>(args...))
			{
				switch (this->m_overlapPolicy)
				{
					case Overlap::ColWise:
						this->apply_colwise(out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
						break;
						
					case Overlap::RowWise:
						this->apply_rowwise(out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
						break;
						
					default:
						SKEPU_ERROR("MapOverlap: Invalid overlap policy");
				}
				return get<0>(args...);
			}
			
		private:
			MapFunc mapFunc;
			MapOverlap1D(MapFunc map): mapFunc(map)
			{
				this->m_edge = Edge::Duplicate;
			}
			
			size_t m_overlap = 0;
			
			friend MapOverlap1D<Ret, Args...> skepu::MapOverlapWrapper<Ret, Args...>(MapFunc);
		};
		
		
		template<typename Ret, typename... Args>
		class MapOverlap2D: public MapOverlapBase<typename region_type<typename pack_element<is_indexed<Args...>::value ? 1 : 0, Args...>::type>::type>
		{
			static constexpr bool indexed = is_indexed<Args...>::value;
			static constexpr size_t InArity = 1;
			static constexpr size_t OutArity = out_size<Ret>::value;
			static constexpr size_t numArgs = sizeof...(Args) - (indexed ? 1 : 0) + OutArity;
			static constexpr size_t anyCont = trait_count_all<is_skepu_container_proxy, Args...>::value;
			
			static constexpr typename make_pack_indices<OutArity, 0>::type out_indices{};
			static constexpr typename make_pack_indices<InArity + OutArity, OutArity>::type elwise_indices{};
			static constexpr typename make_pack_indices<InArity + OutArity + anyCont, InArity + OutArity>::type any_indices{};
			static constexpr typename make_pack_indices<numArgs, InArity + OutArity + anyCont>::type const_indices{};
			
			using MapFunc = std::function<Ret(Args...)>;
			using F = ConditionalIndexForwarder<indexed, MapFunc>;
			using RegionType = typename pack_element<indexed ? 1 : 0, Args...>::type;
			using T = typename region_type<RegionType>::type;
			
		public:
			void setBackend(BackendSpec) {}
			void resetBackend() {}
			
			void setOverlap(size_t o)
			{
				this->m_overlap_i = o;
				this->m_overlap_j = o;
			}
			
			void setOverlap(size_t i, size_t j)
			{
				this->m_overlap_i = i;
				this->m_overlap_j = j;
			}
			
			std::pair<size_t, size_t> getOverlap() const
			{
				return std::make_pair(this->m_overlap_x, this->m_overlap_y);
			}
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void apply(pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
			{
				size_t size_i = get<0>(args...).size_i();
				size_t size_j = get<0>(args...).size_j();
				
				if (disjunction(
					(get<OI>(args...).size_i() < size_i) &&
					(get<OI>(args...).size_j() < size_j) ...))
					SKEPU_ERROR("Non-matching output container sizes");
					
				if (this->m_edge != Edge::None && disjunction(
					(get<EI>(args...).size_i() != size_i) &&
					(get<EI>(args...).size_j() != size_j) ...))
					SKEPU_ERROR("Non-matching input container sizes");
				
				if (this->m_edge == Edge::None && disjunction(
					(get<EI>(args...).size_i() - this->m_overlap_i*2 != size_i) &&
					(get<EI>(args...).size_j() - this->m_overlap_j*2 != size_j) ...))
					SKEPU_ERROR("Non-matching input container sizes");
			
				auto &arg = get<OutArity>(args...);
				
				RegionType region{arg, this->m_overlap_i, this->m_overlap_j, this->m_edge, this->m_pad};
				
				for (size_t i = 0; i < size_i; i++)
					for (size_t j = 0; j < size_j; j++)
					{
						region.idx = (this->m_edge != Edge::None) ? Index2D{i,j} : Index2D{i + this->m_overlap_i, j + this->m_overlap_j};
						auto res = F::forward(this->mapFunc, Index2D{i,j}, region, get<AI>(args...).hostProxy()..., get<CI>(args...)...);
						SKEPU_VARIADIC_RETURN(get<OI>(args...)(i, j)..., res);
					}
			}
			
			template<typename... CallArgs>
			auto operator()(CallArgs&&... args) -> decltype(get<0>(args...))
			{
				apply(out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
				return get<0>(args...);
			}
			
		private:
			MapFunc mapFunc;
			MapOverlap2D(MapFunc map): mapFunc(map)
			{
				this->m_edge = Edge::None;
			}
			
			int m_overlap_i, m_overlap_j;
			
			friend MapOverlap2D<Ret, Args...> skepu::MapOverlapWrapper<Ret, Args...>(MapFunc);
		};
		
		
		
		
		
		
		
		template<typename Ret, typename... Args>
		class MapOverlap3D: public MapOverlapBase<typename region_type<typename pack_element<is_indexed<Args...>::value ? 1 : 0, Args...>::type>::type>
		{
			static constexpr bool indexed = is_indexed<Args...>::value;
			static constexpr size_t InArity = 1;
			static constexpr size_t OutArity = out_size<Ret>::value;
			static constexpr size_t numArgs = sizeof...(Args) - (indexed ? 1 : 0) + OutArity;
			static constexpr size_t anyCont = trait_count_all<is_skepu_container_proxy, Args...>::value;
			
			static constexpr typename make_pack_indices<OutArity, 0>::type out_indices{};
			static constexpr typename make_pack_indices<InArity + OutArity, OutArity>::type elwise_indices{};
			static constexpr typename make_pack_indices<InArity + OutArity + anyCont, InArity + OutArity>::type any_indices{};
			static constexpr typename make_pack_indices<numArgs, InArity + OutArity + anyCont>::type const_indices{};
			
			using MapFunc = std::function<Ret(Args...)>;
			using F = ConditionalIndexForwarder<indexed, MapFunc>;
			using RegionType = typename pack_element<indexed ? 1 : 0, Args...>::type;
			using T = typename region_type<RegionType>::type;
			
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
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void apply(pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
			{
				size_t size_i = get<0>(args...).size_i();
				size_t size_j = get<0>(args...).size_j();
				size_t size_k = get<0>(args...).size_k();
				
				if (disjunction(
					(get<OI>(args...).size_i() < size_i) &&
					(get<OI>(args...).size_j() < size_j) &&
					(get<OI>(args...).size_k() < size_k) ...))
					SKEPU_ERROR("Non-matching output container sizes");
				
				if (this->m_edge != Edge::None && disjunction(
					(get<EI>(args...).size_i() != size_i) &&
					(get<EI>(args...).size_j() != size_j) &&
					(get<EI>(args...).size_k() != size_k) ...))
					SKEPU_ERROR("Non-matching input container sizes");
				
				if (this->m_edge == Edge::None && disjunction(
					(get<EI>(args...).size_i() - this->m_overlap_i*2 != size_i) &&
					(get<EI>(args...).size_j() - this->m_overlap_j*2 != size_j) &&
					(get<EI>(args...).size_k() - this->m_overlap_k*2 != size_k) ...))
					SKEPU_ERROR("Non-matching input container sizes");
			
				auto &arg = get<OutArity>(args...);
				
				RegionType region{arg, this->m_overlap_i, this->m_overlap_j, this->m_overlap_k, this->m_edge, this->m_pad};
				
				for (size_t i = 0; i < size_i; i++)
					for (size_t j = 0; j < size_j; j++)
						for (size_t k = 0; k < size_k; k++)
						{
							region.idx = (this->m_edge != Edge::None) ? Index3D{i,j,k} : Index3D{i + this->m_overlap_i, j + this->m_overlap_j, k + this->m_overlap_k};
							auto res = F::forward(this->mapFunc, Index3D{i,j,k}, region, get<AI>(args...).hostProxy()..., get<CI>(args...)...);
							SKEPU_VARIADIC_RETURN(get<OI>(args...)(i, j, k)..., res);
						}
			}
			
			template<typename... CallArgs>
			auto operator()(CallArgs&&... args) -> decltype(get<0>(args...))
			{
				apply(out_indices, elwise_indices, any_indices, const_indices, args...);
				return get<0>(args...);
			}
			
		private:
			MapFunc mapFunc;
			MapOverlap3D(MapFunc map): mapFunc(map)
			{
				this->m_edge = Edge::None;
			}
			
			int m_overlap_i, m_overlap_j, m_overlap_k;
			
			friend MapOverlap3D<Ret, Args...> skepu::MapOverlapWrapper<Ret, Args...>(MapFunc);
		};
		
		
		
		
		
		template<typename Ret, typename... Args>
		class MapOverlap4D: public MapOverlapBase<typename region_type<typename pack_element<is_indexed<Args...>::value ? 1 : 0, Args...>::type>::type>
		{
			static constexpr bool indexed = is_indexed<Args...>::value;
			static constexpr size_t InArity = 1;
			static constexpr size_t OutArity = out_size<Ret>::value;
			static constexpr size_t numArgs = sizeof...(Args) - (indexed ? 1 : 0) + OutArity;
			static constexpr size_t anyCont = trait_count_all<is_skepu_container_proxy, Args...>::value;
			
			static constexpr typename make_pack_indices<OutArity, 0>::type out_indices{};
			static constexpr typename make_pack_indices<InArity + OutArity, OutArity>::type elwise_indices{};
			static constexpr typename make_pack_indices<InArity + OutArity + anyCont, InArity + OutArity>::type any_indices{};
			static constexpr typename make_pack_indices<numArgs, InArity + OutArity + anyCont>::type const_indices{};
			
			using MapFunc = std::function<Ret(Args...)>;
			using F = ConditionalIndexForwarder<indexed, MapFunc>;
			using RegionType = typename pack_element<indexed ? 1 : 0, Args...>::type;
			using T = typename region_type<RegionType>::type;
			
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
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void apply(pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
			{
				size_t size_i = get<0>(args...).size_i();
				size_t size_j = get<0>(args...).size_j();
				size_t size_k = get<0>(args...).size_k();
				size_t size_l = get<0>(args...).size_l();
				
				if (disjunction(
					(get<OI>(args...).size_i() < size_i) &&
					(get<OI>(args...).size_j() < size_j) &&
					(get<OI>(args...).size_k() < size_k) &&
					(get<OI>(args...).size_l() < size_l)...))
					SKEPU_ERROR("Non-matching output container sizes");
				
				if (this->m_edge != Edge::None && disjunction(
					(get<EI>(args...).size_i() != size_i) &&
					(get<EI>(args...).size_j() != size_j) &&
					(get<EI>(args...).size_k() != size_k) &&
					(get<EI>(args...).size_l() != size_l)...))
					SKEPU_ERROR("Non-matching input container sizes");
				
				if (this->m_edge == Edge::None && disjunction(
					(get<EI>(args...).size_i() - this->m_overlap_i*2 != size_i) &&
					(get<EI>(args...).size_j() - this->m_overlap_j*2 != size_j) &&
					(get<EI>(args...).size_k() - this->m_overlap_k*2 != size_k) &&
					(get<EI>(args...).size_l() - this->m_overlap_l*2 != size_l)...))
					SKEPU_ERROR("Non-matching input container sizes");
			
				auto &arg = get<OutArity>(args...);
				
				RegionType region{arg, this->m_overlap_i, this->m_overlap_j, this->m_overlap_k, this->m_overlap_l, this->m_edge, this->m_pad};
				
				for (size_t i = 0; i < size_i; i++)
					for (size_t j = 0; j < size_j; j++)
						for (size_t k = 0; k < size_k; k++)
							for (size_t l = 0; l < size_l; l++)
							{
								region.idx = (this->m_edge != Edge::None) ? Index4D{i,j,k,l} : Index4D{i + this->m_overlap_i, j + this->m_overlap_j, k + this->m_overlap_k, l + this->m_overlap_l};
								auto res = F::forward(this->mapFunc, Index4D{i,j,k,l}, region, get<AI>(args...).hostProxy()..., get<CI>(args...)...);
								SKEPU_VARIADIC_RETURN(get<OI>(args...)(i, j, k, l)..., res);
							}
			}
			
			template<typename... CallArgs>
			auto operator()(CallArgs&&... args) -> decltype(get<0>(args...))
			{
				this->apply(out_indices, elwise_indices, any_indices, const_indices, args...);
				return get<0>(args...);
			}
			
		private:
			MapFunc mapFunc;
			MapOverlap4D(MapFunc map): mapFunc(map)
			{
				this->m_edge = Edge::None;
			}
			
			int m_overlap_i, m_overlap_j, m_overlap_k, m_overlap_l;
			
			friend MapOverlap4D<Ret, Args...> skepu::MapOverlapWrapper<Ret, Args...>(MapFunc);
		};
		
	}
}
