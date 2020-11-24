
namespace future_std
{
	template<class T>
	#ifdef SKEPU_CUDA
		__host__ __device__
	#endif
	constexpr const T& clamp( const T& v, const T& lo, const T& hi )
	{
	  return (v < lo) ? lo : (hi < v) ? hi : v;
	}
}

namespace skepu
{
	// ----------------------------------------------------------------
	// Region proxy types
	// ----------------------------------------------------------------
	
	template<typename T>
	struct Region1D
	{
		int oi;
		size_t stride;
		const T *data;
		
#ifdef SKEPU_CUDA
	__host__ __device__
#endif
		T operator()(int i)
		{
			return data[i * this->stride];
		}
		
#ifdef SKEPU_CUDA
	__host__ __device__
#endif
		Region1D(int arg_oi, size_t arg_stride, const T *arg_data)
		: oi(arg_oi), stride(arg_stride), data(arg_data) {}
	};
	
	template<typename T>
	struct Region2D
	{
		int oi, oj;
		size_t size_i, size_j;
		size_t stride;
		Index2D idx;
		const T *data;
		Edge edge = Edge::None;
		T pad;
		
#ifdef SKEPU_CUDA
		__host__ __device__
#endif
		T operator()(int i, int j)
		{
			int ii = this->idx.row + i;
			int jj = this->idx.col + j;
			size_t offset;
			
			switch (this->edge)
			{
			case Edge::Cyclic:
				offset =
					((ii + this->size_i) % this->size_i) * this->stride +
					((jj + this->size_j) % this->size_j);
				return data[offset];
			case Edge::Pad:
				if ((0 <= ii && ii < size_i) && (0 <= jj && jj < size_j))
					return data[ii * this->stride + jj];
				else return this->pad;
			case Edge::Duplicate:
				offset =
					future_std::clamp<int>(ii, 0, this->size_i-1) * this->stride +
					future_std::clamp<int>(jj, 0, this->size_j-1);
				return data[offset];
			default:
				return data[ii * this->stride + jj];
			}
		}
		
		Region2D(Matrix<T> &mat, int arg_oi, int arg_oj, Edge arg_edge, T arg_pad)
		:	oi(arg_oi), oj(arg_oj),
			size_i(mat.size_i()), size_j(mat.size_j()),
			stride(this->size_j),
			edge(arg_edge),
			pad(arg_pad),
			data(mat.getAddress())
		{}

// For CUDA kernel
#ifdef SKEPU_CUDA
		__host__ __device__
#endif
		Region2D(int arg_oj, int arg_oi, size_t arg_stride, T *arg_data)
		:	oi(arg_oi), oj(arg_oj),
			size_i(static_cast<size_t>(-1)), size_j(static_cast<size_t>(-1)),
			stride(arg_stride),
			edge(Edge::None),
			pad(0),
			data(arg_data),
			idx{0,0}
		{}
	};
	
	template<typename T>
	struct Region3D
	{
		int oi, oj, ok;
		size_t size_i, size_j, size_k;
		size_t stride1, stride2;
		Index3D idx;
		const T *data;
		Edge edge = Edge::None;
		T pad;
		
#ifdef SKEPU_CUDA
		__host__ __device__
#endif
		T operator()(int i, int j, int k)
		{
			int ii = this->idx.i + i;
			int jj = this->idx.j + j;
			int kk = this->idx.k + k;
			size_t offset;
			
			switch (this->edge)
			{
			case Edge::Cyclic:
				offset =
					((ii + this->size_i) % this->size_i) * this->stride1 +
					((jj + this->size_j) % this->size_j) * this->stride2 +
					((kk + this->size_k) % this->size_k);
				return data[offset];
			case Edge::Pad:
				if ((0 <= ii && ii < size_i) && (0 <= jj && jj < size_j) &&
						(0 <= kk && kk < size_k))
					return data[ii * this->stride1 + jj * this->stride2 + kk];
				else return this->pad;
			case Edge::Duplicate:
				offset =
					future_std::clamp<int>(ii, 0, this->size_i-1) * this->stride1 +
					future_std::clamp<int>(jj, 0, this->size_j-1) * this->stride2 +
					future_std::clamp<int>(kk, 0, this->size_k-1);
				return data[offset];
			default:
				return data[ii * this->stride1 + jj * this->stride2 + kk];
			}
		}
		
#ifdef SKEPU_CUDA
		__host__ __device__
#endif
		Region3D(Tensor3<T> &ten, int arg_oi, int arg_oj, int arg_ok, Edge arg_edge, T arg_pad)
		:	oi(arg_oi), oj(arg_oj), ok(arg_ok),
			size_i(ten.size_i()), size_j(ten.size_j()), size_k(ten.size_k()),
			stride1(ten.m_stride_1),
			stride2(ten.m_stride_2),
			edge(arg_edge),
			pad(arg_pad),
			data(ten.getAddress())
		{}
	};
	
	template<typename T>
	struct Region4D
	{
		int oi, oj, ok, ol;
		size_t size_i, size_j, size_k, size_l;
		size_t stride1, stride2, stride3;
		Index4D idx;
		const T *data;
		Edge edge = Edge::None;
		T pad;
		
#ifdef SKEPU_CUDA
		__host__ __device__
#endif
		T operator()(int i, int j, int k, int l)
		{
			int ii = this->idx.i + i;
			int jj = this->idx.j + j;
			int kk = this->idx.k + k;
			int ll = this->idx.l + l;
			size_t offset;
			
			switch (this->edge)
			{
			case Edge::Cyclic:
				offset =
					((ii + this->size_i) % this->size_i) * this->stride1 +
					((jj + this->size_j) % this->size_j) * this->stride2 +
					((kk + this->size_k) % this->size_k) * this->stride3 +
					((ll + this->size_l) % this->size_l);
				return data[offset];
			case Edge::Pad:
				if ((0 <= ii && ii < size_i) && (0 <= jj && jj < size_j) &&
					  (0 <= kk && kk < size_k) && (0 <= ll && ll < size_l))
					return data[ii * this->stride1 + jj * this->stride2 + kk * this->stride3 + ll];
				else return this->pad;
			case Edge::Duplicate:
				offset =
					future_std::clamp<int>(ii, 0, this->size_i-1) * this->stride1 +
					future_std::clamp<int>(jj, 0, this->size_j-1) * this->stride2 +
					future_std::clamp<int>(kk, 0, this->size_k-1) * this->stride3 +
					future_std::clamp<int>(ll, 0, this->size_l-1);
				return data[offset];
			default:
				return data[ii * this->stride1 + jj * this->stride2 + kk * this->stride3 + ll];
			}
		}
		
#ifdef SKEPU_CUDA
		__host__ __device__
#endif
		Region4D(Tensor4<T> &ten, int arg_oi, int arg_oj, int arg_ok, int arg_ol, Edge arg_edge, T arg_pad)
		:	oi(arg_oi), oj(arg_oj), ok(arg_ok), ol(arg_ol),
			size_i(ten.size_i()), size_j(ten.size_j()), size_k(ten.size_k()), size_l(ten.size_l()),
			stride1(ten.m_stride_1),
			stride2(ten.m_stride_2),
			stride3(ten.m_stride_3),
			edge(arg_edge),
			pad(arg_pad),
			data(ten.getAddress())
		{}
	};
	
	
	template<typename T>
	std::ostream& operator<<(std::ostream &os, Region1D<T>& r)
	{
		os << "Region1: (" << (2 * r.oi + 1) << ")\n";
		
		for (int i = -r.oi; i <= r.oi; ++i)
		{
			auto val = r(i);
			std::cout << std::setw(5) << val << " ";
		}
		return os << "\n";
	}
	
	template<typename T>
	std::ostream& operator<<(std::ostream &os, Region2D<T>& r)
	{
		os << "Region2: (" << (2 * r.oi + 1)
			<< " x " << (2 * r.oj + 1) << ")\n";
		
		for (int i = -r.oi; i <= r.oi; ++i)
		{
			for (int j = -r.oj; j <= r.oj; ++j)
			{
				auto val = r(i,j);
				std::cout << std::setw(5) << val << " ";
			}
			os << "\n";
		}
		return os << "\n";
	}
	
	template<typename T>
	std::ostream& operator<<(std::ostream &os, Region3D<T>& r)
	{
		os << "Region3: (" << (2 * r.oi + 1)
			<< " x " << (2 * r.oj + 1)
			<< " x " << (2 * r.ok + 1) << ")\n";
		
		for (int j = -r.oj; j <= r.oj; ++j)
		{
			for (int i = -r.oi; i <= r.oi; ++i)
			{
				for (int k = -r.ok; k <= r.ok; ++k)
				{
					auto val = r(i,j,k);
					std::cout << std::setw(5) << val << " ";
				}
				os << " | ";
			}
			os << "\n";
		}
		return os << "\n";
	}
	
	template<typename T>
	std::ostream& operator<<(std::ostream &os, Region4D<T>& r)
	{
		os << "Region4: (" << (2 * r.oi + 1)
			<< " x " << (2 * r.oj + 1)
			<< " x " << (2 * r.ok + 1)
			<< " x " << (2 * r.ol + 1) << ")\n";
		
		for (int i = -r.oi; i <= r.oi; ++i)
		{
			for (int k = -r.ok; k <= r.ok; ++k)
			{
				for (int j = -r.oj; j <= r.oj; ++j)
				{
					for (int l = -r.ol; l <= r.ol; ++l)
					{
						auto val = r(i,j,k,l);
						std::cout << std::setw(5) << val << " ";
					}
					os << " | ";
				}
				os << "\n";
			}
			os << "---------------------------------\n";
		}
		return os << "\n";
	}
	
	
	// ----------------------------------------------------------------
	// MapOverlap type deducer
	// ----------------------------------------------------------------
	
	template<typename T>
	struct region_type {};
	
	template<typename T>
	struct region_type<Region1D<T>> { using type = T; };
	
	template<typename T>
	struct region_type<Region2D<T>> { using type = T; };
	
	template<typename T>
	struct region_type<Region3D<T>> { using type = T; };
	
	template<typename T>
	struct region_type<Region4D<T>> { using type = T; };
	
	
	// ----------------------------------------------------------------
	// MapOverlap dimensionality deducer
	// ----------------------------------------------------------------
	
	template<typename... Args>
	struct mapoverlap_dimensionality {};
	
	template<typename T, typename... Args>
	struct mapoverlap_dimensionality<Region1D<T>, Args...>: std::integral_constant<int, 1> {};
	
	template<typename T, typename... Args>
	struct mapoverlap_dimensionality<Region2D<T>, Args...>: std::integral_constant<int, 2> {};
	
	template<typename T, typename... Args>
	struct mapoverlap_dimensionality<Region3D<T>, Args...>: std::integral_constant<int, 3> {};
	
	template<typename T, typename... Args>
	struct mapoverlap_dimensionality<Region4D<T>, Args...>: std::integral_constant<int, 4> {};
	
	template<typename T, typename... Args>
	struct mapoverlap_dimensionality<Index1D, Region1D<T>, Args...>: std::integral_constant<int, 1> {};
	
	template<typename T, typename... Args>
	struct mapoverlap_dimensionality<Index2D, Region2D<T>, Args...>: std::integral_constant<int, 2> {};
	
	template<typename T, typename... Args>
	struct mapoverlap_dimensionality<Index3D, Region3D<T>, Args...>: std::integral_constant<int, 3> {};
	
	template<typename T, typename... Args>
	struct mapoverlap_dimensionality<Index4D, Region4D<T>, Args...>: std::integral_constant<int, 4> {};
	

}