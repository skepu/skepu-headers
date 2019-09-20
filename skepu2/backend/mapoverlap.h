/*! \file mapoverlap.h
 *  \brief Contains a class declaration for the MapOverlap skeleton.
 */

#ifndef MAPOVERLAP_H
#define MAPOVERLAP_H

namespace skepu2
{
	/*!
	 *  Enumeration of the different edge policies (what happens when a read outside the vector is performed) that the map overlap skeletons support.
	 */
	enum class Edge
	{
		Pad = 0, Cyclic = 1, Duplicate = 2
	};
	
	enum class Overlap
	{
		RowWise, ColWise, RowColWise, ColRowWise
	};
	
	namespace backend
	{
		/*!
		 *  \ingroup skeletons
		 */
		/*!
		 *  \class MapOverlap
		 *
		 *  \brief A class representing the MapOverlap skeleton.
		 *
		 *  This class defines the MapOverlap skeleton which is similar to a Map, but each element of the result (vecor/matrix) is a function
		 *  of \em several adjacent elements of one input (vecor/matrix) that reside at a certain constant maximum distance from each other.
		 *  This class can be used to apply (1) overlap to a vector and (2) separable-overlap to a matrix (row-wise, column-wise). For
		 *  non-separable matrix overlap which considers diagonal neighbours as well besides row- and column-wise neighbours, please see \p src/MapOverlap2D.
		 *  MapOverlap2D class can be used by including same header file (i.e., mapoverlap.h) but class name is different (MapOverlap2D).
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel>
		class MapOverlap1D: public SkeletonBase
		{
			using Ret = typename MapOverlapFunc::Ret;
			using T = typename std::remove_cv<typename std::remove_pointer<typename parameter_type<2, decltype(&MapOverlapFunc::CPU)>::type>::type>::type;
			
		public:
			
			static constexpr auto skeletonType = SkeletonType::MapOverlap1D;
			using ResultArg = std::tuple<T>;
			using ElwiseArgs = std::tuple<T>;
			using ContainerArgs = typename MapOverlapFunc::ContainerArgs;
			using UniformArgs = typename MapOverlapFunc::UniformArgs;
			static constexpr bool prefers_matrix = false;
			
			MapOverlap1D(CUDAKernel kernel, C2 k2, C3 k3, C4 k4)
			: m_cuda_kernel(kernel), m_cuda_rowwise_kernel(k2), m_cuda_colwise_kernel(k3), m_cuda_colwise_multi_kernel(k4)
			{
#ifdef SKEPU_OPENCL
				CLKernel::initialize();
#endif
			}
			
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
			
			void setOverlap(size_t o)
			{
				this->m_overlap = o;
			}
			
			size_t getOverlap() const
			{
				return this->m_overlap;
			}
			
			template<typename... Args>
			void tune(Args&&... args)
			{
				tuner::tune(*this, std::forward<Args>(args)...);
			}
			
		private:
			CUDAKernel m_cuda_kernel;
			C2 m_cuda_rowwise_kernel;
			C3 m_cuda_colwise_kernel;
			C4 m_cuda_colwise_multi_kernel;
			
			Overlap m_overlapPolicy = Overlap::RowWise;
			Edge m_edge = Edge::Duplicate;
			T m_pad {};
			
			size_t m_overlap;
		
		public:
		   
		
		private:
			template<template<class> class Container, size_t... AnyIndx, size_t... ConstIndx, typename... CallArgs>
			void vector_CPU(Container<Ret>& res, Container<T>& arg, pack_indices<AnyIndx...>, pack_indices<ConstIndx...>, CallArgs&&... args);
			
			template<size_t... AnyIndx, size_t... ConstIndx, typename... CallArgs>
			void rowwise_CPU(Matrix<Ret>& res, Matrix<T>& arg, pack_indices<AnyIndx...>, pack_indices<ConstIndx...>, CallArgs&&... args);
			
			template<size_t... AnyIndx, size_t... ConstIndx, typename... CallArgs>
			void colwise_CPU(Matrix<Ret>& res, Matrix<T>& arg, pack_indices<AnyIndx...>, pack_indices<ConstIndx...>, CallArgs&&... args);
		   
		
#ifdef SKEPU_OPENMP
		private:
			template<template<class> class Container, size_t... AnyIndx, size_t... ConstIndx, typename... CallArgs>
			void vector_OpenMP(Container<Ret>& res, Container<T>& arg, pack_indices<AnyIndx...>, pack_indices<ConstIndx...>, CallArgs&&... args);
			
			template<size_t... AnyIndx, size_t... ConstIndx, typename... CallArgs>
			void rowwise_OpenMP(Matrix<Ret>& res, Matrix<T>& arg, pack_indices<AnyIndx...>, pack_indices<ConstIndx...>, CallArgs&&... args);
			
			template<size_t... AnyIndx, size_t... ConstIndx, typename... CallArgs>
			void colwise_OpenMP(Matrix<Ret>& res, Matrix<T>& arg, pack_indices<AnyIndx...>, pack_indices<ConstIndx...>, CallArgs&&... args);
		
#endif
		
#ifdef SKEPU_CUDA
		public:
		   
		
		private:
			template<template<class> class Container, size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapSingleThread_CU(size_t deviceID, size_t startIdx, Container<Ret>& res, Container<T>& arg, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args);
			
			template<template<class> class Container, size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapNumDevices_CU(size_t numDevices, size_t startIdx, Container<Ret>& res, Container<T>& arg, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args);
			
			template<template<class> class Container, size_t... AI, size_t... CI, typename... CallArgs>
			void vector_CUDA(size_t startIdx, Container<Ret>& res, Container<T>& arg, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args);
			
			
			template<size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapSingleThread_CU_Col(size_t deviceID, size_t numcols, Matrix<Ret>& res, Matrix<T>& arg, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args);
			
			template<size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapMultiThread_CU_Col(size_t numDevices, size_t numcols, Matrix<Ret>& res, Matrix<T>& arg, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args);
			
			template<size_t... AI, size_t... CI, typename... CallArgs>
			void colwise_CUDA(size_t numcols, Matrix<Ret>& res, Matrix<T>& arg, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args);
			
			
			template<size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapSingleThread_CU_Row(size_t deviceID, size_t numrows, Matrix<Ret>& res, Matrix<T>& arg, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args);
			
			template<size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapMultiThread_CU_Row(size_t numDevices, size_t numrows, Matrix<Ret>& res, Matrix<T>& arg, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args);
			
			template<size_t... AI, size_t... CI, typename... CallArgs>
			void rowwise_CUDA(size_t numrows, Matrix<Ret>& res, Matrix<T>& arg, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args);
			
			template <typename T>
			size_t getThreadNumber_CU(size_t width, size_t &numThreads, size_t deviceID);
			
			template <typename T>
			bool sharedMemAvailable_CU(size_t &numThreads, size_t deviceID);
			
#endif
		
#ifdef SKEPU_OPENCL
		private:
			template<template<class> class Container, size_t... AnyIndx, size_t... ConstIndx, typename... CallArgs>
			void vector_OpenCL(size_t startIdx, Container<Ret>& res, Container<T>& arg, pack_indices<AnyIndx...>, pack_indices<ConstIndx...>, CallArgs&&... args);
			
			template<size_t... AnyIndx, size_t... ConstIndx, typename... CallArgs>
			void rowwise_OpenCL(size_t numrows, Matrix<Ret>& res, Matrix<T>& arg, pack_indices<AnyIndx...>, pack_indices<ConstIndx...>, CallArgs&&... args);
			
			template<size_t... AnyIndx, size_t... ConstIndx, typename... CallArgs>
			void colwise_OpenCL(size_t numcols, Matrix<Ret>& res, Matrix<T>& arg, pack_indices<AnyIndx...>, pack_indices<ConstIndx...>, CallArgs&&... args);
		
			
			template<template<class> class Container, size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapSingle_CL(size_t deviceID, size_t startIdx, Container<Ret>& res, Container<T>& arg, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args);
			
			template<template<class> class Container, size_t... AnyIndx, size_t... ConstIndx, typename... CallArgs>
			void mapOverlapNumDevices_CL(size_t numDevices, size_t startIdx, Container<Ret>& res, Container<T>& arg, pack_indices<AnyIndx...> ai, pack_indices<ConstIndx...> ci, CallArgs&&... args);
			
			template<size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapSingle_CL_Row(size_t deviceID, size_t numrows, Matrix<Ret>& res, Matrix<T>& arg, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args);
			
			template<size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapSingle_CL_RowMulti(size_t numDevices, size_t numrows, Matrix<Ret>& res, Matrix<T>& arg, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args);
			
			template<size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapSingle_CL_Col(size_t deviceID, size_t numcols, Matrix<Ret>& res, Matrix<T>& arg, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args);
			
			template<size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapSingle_CL_ColMulti(size_t numDevices, size_t numcols, Matrix<Ret>& res, Matrix<T>& arg, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args);
			
			
			template<typename T>
			int getThreadNumber_CL(size_t width, size_t numThreads, size_t deviceID);
			
			template<typename T>
			bool sharedMemAvailable_CL(size_t &numThreads, size_t deviceID);
			
#endif
			
			
		
#ifdef SKEPU_HYBRID
		private:
			template<template<class> class Container, size_t... AnyIndx, size_t... ConstIndx, typename... CallArgs>
			void vector_Hybrid(Container<Ret>& res, Container<T>& arg, pack_indices<AnyIndx...>, pack_indices<ConstIndx...>, CallArgs&&... args);
			
			template<size_t... AnyIndx, size_t... ConstIndx, typename... CallArgs>
			void rowwise_Hybrid(Matrix<Ret>& res, Matrix<T>& arg, pack_indices<AnyIndx...>, pack_indices<ConstIndx...>, CallArgs&&... args);
			
			template<size_t... AnyIndx, size_t... ConstIndx, typename... CallArgs>
			void colwise_Hybrid(Matrix<Ret>& res, Matrix<T>& arg, pack_indices<AnyIndx...>, pack_indices<ConstIndx...>, CallArgs&&... args);
		
#endif
		
		public:
			template<template<class> class Container, typename... CallArgs>
			Container<Ret> &operator()(Container<Ret> &res, Container<T> &arg, CallArgs&&... args)
			{
				static constexpr size_t anyCont = std::tuple_size<typename MapOverlapFunc::ContainerArgs>::value;
				typename make_pack_indices<anyCont, 0>::type any_indices;
				typename make_pack_indices<sizeof...(CallArgs), anyCont>::type const_indices;
				
				this->m_selected_spec = (this->m_user_spec != nullptr)
					? this->m_user_spec
					: &this->m_execPlan->find(arg.size());
				
				switch (this->m_selected_spec->backend())
				{
				case Backend::Type::Hybrid:
#ifdef SKEPU_HYBRID
					this->vector_Hybrid(res, arg, any_indices, const_indices, std::forward<CallArgs>(args)...);
					break;
#endif
				case Backend::Type::CUDA:
#ifdef SKEPU_CUDA
					this->vector_CUDA(0, res, arg, any_indices, const_indices, std::forward<CallArgs>(args)...);
					break;
#endif
				case Backend::Type::OpenCL:
#ifdef SKEPU_OPENCL
					this->vector_OpenCL(0, res, arg, any_indices, const_indices, std::forward<CallArgs>(args)...);
					break;
#endif
				case Backend::Type::OpenMP:
#ifdef SKEPU_OPENMP
					this->vector_OpenMP(res, arg, any_indices, const_indices, std::forward<CallArgs>(args)...);
					break;
#endif
				default:
					this->vector_CPU(res, arg, any_indices, const_indices, std::forward<CallArgs>(args)...);
					break;
				}
				
				return res;
			}
			
			template<typename... CallArgs>
			Matrix<Ret> &operator()(Matrix<Ret> &res, Matrix<T> &arg, CallArgs&&... args)
			{
				if (arg.total_rows() != res.total_rows() || arg.total_cols() != res.total_cols())
					SKEPU_ERROR("MapOverlap 1D: Non-matching container sizes");
				
				static constexpr size_t anyCont = std::tuple_size<typename MapOverlapFunc::ContainerArgs>::value;
				typename make_pack_indices<anyCont, 0>::type any_indices;
				typename make_pack_indices<sizeof...(CallArgs), anyCont>::type const_indices;
				
				this->m_selected_spec = (this->m_user_spec != nullptr)
					? this->m_user_spec
					: &this->m_execPlan->find(arg.size());
				
				size_t numrows = arg.total_rows();
				size_t numcols = arg.total_cols();
					
				switch (this->m_overlapPolicy)
				{
					case Overlap::RowColWise: {
						Matrix<Ret> tmp(res.total_rows(), res.total_cols());
						switch (this->m_selected_spec->backend())
						{
						case Backend::Type::Hybrid:
#ifdef SKEPU_HYBRID
							this->rowwise_Hybrid(tmp, arg, any_indices, const_indices, std::forward<CallArgs>(args)...);
							this->colwise_Hybrid(res, tmp, any_indices, const_indices, std::forward<CallArgs>(args)...);
							break;
#endif
						case Backend::Type::CUDA:
#ifdef SKEPU_CUDA
							this->rowwise_CUDA(numrows, tmp, arg, any_indices, const_indices, std::forward<CallArgs>(args)...);
							this->colwise_CUDA(numcols, res, tmp, any_indices, const_indices, std::forward<CallArgs>(args)...);
							break;
#endif
						case Backend::Type::OpenCL:
#ifdef SKEPU_OPENCL
							this->rowwise_OpenCL(numrows, tmp, arg, any_indices, const_indices, std::forward<CallArgs>(args)...);
							this->colwise_OpenCL(numcols, res, tmp, any_indices, const_indices, std::forward<CallArgs>(args)...);
							break;
#endif
						case Backend::Type::OpenMP:
#ifdef SKEPU_OPENMP
							this->rowwise_OpenMP(tmp, arg, any_indices, const_indices, std::forward<CallArgs>(args)...);
							this->colwise_OpenMP(res, tmp, any_indices, const_indices, std::forward<CallArgs>(args)...);
							break;
#endif
						default:
							this->rowwise_CPU(tmp, arg, any_indices, const_indices, std::forward<CallArgs>(args)...);
							this->colwise_CPU(res, tmp, any_indices, const_indices, std::forward<CallArgs>(args)...);
							break;
						}
						break;
					}
					
					case Overlap::ColRowWise: {
						Matrix<Ret> tmp(res.total_rows(), res.total_cols());
						switch (this->m_selected_spec->backend())
						{
						case Backend::Type::Hybrid:
#ifdef SKEPU_HYBRID
							this->colwise_Hybrid(tmp, arg, any_indices, const_indices, std::forward<CallArgs>(args)...);
							this->rowwise_Hybrid(res, tmp, any_indices, const_indices, std::forward<CallArgs>(args)...);
							break;
#endif
						case Backend::Type::CUDA:
#ifdef SKEPU_CUDA
							this->colwise_CUDA(numcols, tmp, arg, any_indices, const_indices, std::forward<CallArgs>(args)...);
							this->rowwise_CUDA(numrows, res, tmp, any_indices, const_indices, std::forward<CallArgs>(args)...);
							break;
#endif
						case Backend::Type::OpenCL:
#ifdef SKEPU_OPENCL
							this->colwise_OpenCL(numcols, tmp, arg, any_indices, const_indices, std::forward<CallArgs>(args)...);
							this->rowwise_OpenCL(numrows, res, tmp, any_indices, const_indices, std::forward<CallArgs>(args)...);
							break;
#endif
						case Backend::Type::OpenMP:
#ifdef SKEPU_OPENMP
							this->colwise_OpenMP(tmp, arg, any_indices, const_indices, std::forward<CallArgs>(args)...);
							this->rowwise_OpenMP(res, tmp, any_indices, const_indices, std::forward<CallArgs>(args)...);
							break;
#endif
						default:
							this->colwise_CPU(tmp, arg, any_indices, const_indices, std::forward<CallArgs>(args)...);
							this->rowwise_CPU(res, tmp, any_indices, const_indices, std::forward<CallArgs>(args)...);
							break;
						}
						break;
					}
					
					case Overlap::ColWise:
						switch (this->m_selected_spec->backend())
						{
						case Backend::Type::Hybrid:
#ifdef SKEPU_HYBRID
							this->colwise_Hybrid(res, arg, any_indices, const_indices, std::forward<CallArgs>(args)...);
							break;
#endif
						case Backend::Type::CUDA:
#ifdef SKEPU_CUDA
							this->colwise_CUDA(numcols, res, arg, any_indices, const_indices, std::forward<CallArgs>(args)...);
							break;
#endif
						case Backend::Type::OpenCL:
#ifdef SKEPU_OPENCL
							this->colwise_OpenCL(numcols, res, arg, any_indices, const_indices, std::forward<CallArgs>(args)...);
							break;
#endif
						case Backend::Type::OpenMP:
#ifdef SKEPU_OPENMP
							this->colwise_OpenMP(res, arg, any_indices, const_indices, std::forward<CallArgs>(args)...);
							break;
#endif
						default:
							this->colwise_CPU(res, arg, any_indices, const_indices, std::forward<CallArgs>(args)...);
							break;
						}
						break;
					
					case Overlap::RowWise:
						switch (this->m_selected_spec->backend())
						{
						case Backend::Type::Hybrid:
#ifdef SKEPU_HYBRID
							this->rowwise_Hybrid(res, arg, any_indices, const_indices, std::forward<CallArgs>(args)...);
							break;
#endif
						case Backend::Type::CUDA:
#ifdef SKEPU_CUDA
							this->rowwise_CUDA(numrows, res, arg, any_indices, const_indices, std::forward<CallArgs>(args)...);
							break;
#endif
						case Backend::Type::OpenCL:
#ifdef SKEPU_OPENCL
							this->rowwise_OpenCL(numrows, res, arg, any_indices, const_indices, std::forward<CallArgs>(args)...);
							break;
#endif
						case Backend::Type::OpenMP:
#ifdef SKEPU_OPENMP
							this->rowwise_OpenMP(res, arg, any_indices, const_indices, std::forward<CallArgs>(args)...);
							break;
#endif
						default:
							this->rowwise_CPU(res, arg, any_indices, const_indices, std::forward<CallArgs>(args)...);
							break;
						}
						break;
						
					default:
						SKEPU_ERROR("MapOverlap: Invalid overlap mode");
				}
				
				return res;
			}
		};
		
		
		template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel>
		class MapOverlap2D: public SkeletonBase
		{
			using Ret = typename MapOverlapFunc::Ret;
			using T = typename std::remove_cv<typename std::remove_pointer<typename parameter_type<3, decltype(&MapOverlapFunc::CPU)>::type>::type>::type;
			
		public:
			
			static constexpr auto skeletonType = SkeletonType::MapOverlap2D;
			using ResultArg = std::tuple<T>;
			using ElwiseArgs = std::tuple<T>;
			using ContainerArgs = typename MapOverlapFunc::ContainerArgs;
			using UniformArgs = typename MapOverlapFunc::UniformArgs;
			static constexpr bool prefers_matrix = true;
			
			MapOverlap2D(CUDAKernel kernel) : m_cuda_kernel(kernel)
			{
#ifdef SKEPU_OPENCL
				CLKernel::initialize();
#endif
			}
			
			void setEdgeMode(Edge mode)
			{
				this->m_edge = mode;
			}
			
			void setPad(T pad)
			{
				this->m_pad = pad;
			}
			
			void setOverlap(size_t o)
			{
				this->m_overlap_x = o;
				this->m_overlap_y = o;
			}
			
			void setOverlap(size_t y, size_t x)
			{
				this->m_overlap_x = x;
				this->m_overlap_y = y;
			}
			
			std::pair<size_t, size_t> getOverlap() const
			{
				return std::make_pair(this->m_overlap_x, this->m_overlap_y);
			}
			
			template<typename... Args>
			void tune(Args&&... args)
			{
				tuner::tune(*this, std::forward<Args>(args)...);
			}
			
		private:
			CUDAKernel m_cuda_kernel;
			
			Edge m_edge = Edge::Duplicate;
			T m_pad {};
			
			size_t m_overlap_x, m_overlap_y;
			
			
		private:
			template<size_t... AnyIndx, size_t... ConstIndx, typename... CallArgs>
			void helper_CPU(Matrix<Ret>& res, Matrix<T>& arg, pack_indices<AnyIndx...>, pack_indices<ConstIndx...>,  CallArgs&&... args);
			
#ifdef SKEPU_OPENMP
			
			template<size_t... AnyIndx, size_t... ConstIndx, typename... CallArgs>
			void helper_OpenMP(Matrix<Ret>& res, Matrix<T>& arg, pack_indices<AnyIndx...>, pack_indices<ConstIndx...>,  CallArgs&&... args);
			
#endif
		
#ifdef SKEPU_OPENCL
			
			template<size_t... AI, size_t... CI, typename... CallArgs>
			void helper_OpenCL(skepu2::Matrix<Ret>& res, skepu2::Matrix<T>& arg, pack_indices<AI...>, pack_indices<CI...>,  CallArgs&&... args);
			
			template<size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapSingleThread_CL(size_t deviceID, skepu2::Matrix<Ret>& res, skepu2::Matrix<T>& arg, pack_indices<AI...>, pack_indices<CI...>,  CallArgs&&... args);
			
			template<size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapMultipleThread_CL(size_t numDevices, skepu2::Matrix<Ret>& res, skepu2::Matrix<T>& arg, pack_indices<AI...>, pack_indices<CI...>,  CallArgs&&... args);
			
#endif
		
#ifdef SKEPU_CUDA
			
			template<size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapSingleThread_CU(size_t deviceID, Matrix<Ret>& res, Matrix<T>& arg, pack_indices<AI...>, pack_indices<CI...>,  CallArgs&&... args);
			
			template<size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapMultipleThread_CU(size_t numDevices, Matrix<Ret>& res, Matrix<T>& arg, pack_indices<AI...>, pack_indices<CI...>,  CallArgs&&... args);
			
			template<size_t... AI, size_t... CI, typename... CallArgs>
			void helper_CUDA(skepu2::Matrix<Ret>& res, skepu2::Matrix<T>& arg, pack_indices<AI...> ai, pack_indices<CI...> ci,  CallArgs&&... args);
			
#endif
			
#ifdef SKEPU_HYBRID
			
			template<size_t... AnyIndx, size_t... ConstIndx, typename... CallArgs>
			void helper_Hybrid(Matrix<Ret>& res, Matrix<T>& arg, pack_indices<AnyIndx...>, pack_indices<ConstIndx...>,  CallArgs&&... args);
			
#endif
			
		public:
			template<typename... CallArgs>
			Matrix<Ret> &operator()(Matrix<Ret> &res, Matrix<T> &arg, CallArgs&&... args)
			{
				static constexpr size_t anyCont = std::tuple_size<typename MapOverlapFunc::ContainerArgs>::value;
				typename make_pack_indices<anyCont, 0>::type any_indices;
				typename make_pack_indices<sizeof...(CallArgs), anyCont>::type const_indices;
				
				const size_t overlap_x = this->m_overlap_x;
				const size_t overlap_y = this->m_overlap_y;
				const size_t in_rows = arg.total_rows();
				const size_t in_cols = arg.total_cols();
				const size_t out_rows = res.total_rows();
				const size_t out_cols = res.total_cols();
				
				if ((in_rows - overlap_y*2 != out_rows) && (in_cols - overlap_x*2 != out_cols))
					SKEPU_ERROR("MapOverlap 2D: Non-matching container sizes");
				
				this->m_selected_spec = (this->m_user_spec != nullptr)
					? this->m_user_spec
					: &this->m_execPlan->find(arg.size());
				
				switch (this->m_selected_spec->backend())
				{
				case Backend::Type::Hybrid:
#ifdef SKEPU_HYBRID
					this->helper_OpenMP(res, arg, any_indices, const_indices, std::forward<CallArgs>(args)...);
					break;
#endif
				case Backend::Type::CUDA:
#ifdef SKEPU_CUDA
					this->helper_CUDA(res, arg, any_indices, const_indices, std::forward<CallArgs>(args)...);
					break;
#endif
				case Backend::Type::OpenCL:
#ifdef SKEPU_OPENCL
					this->helper_OpenCL(res, arg, any_indices, const_indices, std::forward<CallArgs>(args)...);
					break;
#endif
				case Backend::Type::OpenMP:
#ifdef SKEPU_OPENMP
					this->helper_OpenMP(res, arg, any_indices, const_indices, std::forward<CallArgs>(args)...);
					break;
#endif
				default:
					this->helper_CPU(res, arg, any_indices, const_indices, std::forward<CallArgs>(args)...);
					break;
				}
				
				return res;
			}
		};
		
	} // namespace backend

#ifdef SKEPU_MERCURIUM

template<typename Ret, typename Arg1, typename... Args>
class MapOverlapImpl: public SeqSkeletonBase
{
protected:
	using T =
		typename std::remove_const<typename std::remove_pointer<Arg1>::type>::type;
	using MapFunc1D = std::function<Ret(int, size_t, Arg1, Args...)>;
	using MapFunc2D = std::function<Ret(int, int, size_t, Arg1, Args...)>;

public:
	void setOverlapMode(Overlap mode);
	void setEdgeMode(Edge mode);
	void setPad(T pad);

	MapOverlapImpl(MapFunc1D map);
	MapOverlapImpl(MapFunc2D map);

	void setOverlap(size_t o);
	void setOverlap(size_t y, size_t x);
	size_t getOverlap() const;
	std::pair<size_t, size_t> getOverlap() const;

	template<
		template<class> class Container,
		size_t... AI,
		size_t... CI,
		typename... CallArgs>
	Container<Ret> &helper(
		Container<Ret> &res,
		Container<T> &arg,
		pack_indices<AI...>,
		pack_indices<CI...>,
		CallArgs&&... args);

	template<template<class> class Container, typename... CallArgs>
	Container<Ret> &operator()(
		Container<Ret> &res, Container<T>& arg, CallArgs&&... args);

	template<size_t... AI, size_t... CI, typename... CallArgs>
	void apply_colwise(
		skepu2::Matrix<Ret>& res,
		skepu2::Matrix<T>& arg,
		pack_indices<AI...>,
		pack_indices<CI...>,
		CallArgs&&... args);

	template<size_t... AI, size_t... CI, typename... CallArgs>
	void apply_rowwise(
		skepu2::Matrix<Ret>& res,
		skepu2::Matrix<T>& arg,
		pack_indices<AI...>,
		pack_indices<CI...>,
		CallArgs&&... args);

	template<typename... CallArgs>
	Matrix<Ret> &operator()(Matrix<Ret> &res, Matrix<T>& arg, CallArgs&&... args);

	template<size_t... AI, size_t... CI, typename... CallArgs>
	void apply_helper(
		Matrix<Ret> &res,
		Matrix<T> &arg,
		pack_indices<AI...>,
		pack_indices<CI...>,
		CallArgs&&... args);

	template<typename... CallArgs>
	Matrix<Ret> &operator()(Matrix<Ret> &res, Matrix<T>& arg, CallArgs&&... args);

};

template<typename Ret, typename Arg1, typename ... ArgRest>
auto inline
MapOverlap(Ret (*)(int, size_t, Arg1, ArgRest...))
-> MapOverlapImpl<Ret, Arg1, ArgRest...>;

template<typename Ret, typename Arg1, typename ... ArgRest>
auto inline
MapOverlap(Ret (*)(int, int, size_t, Arg1, ArgRest...))
-> MapOverlapImpl<Ret, Arg1, ArgRest...>;

#endif // SKEPU_MERCURIUM

} // namespace skepu2


#include "impl/mapoverlap/mapoverlap_cpu.inl"
#include "impl/mapoverlap/mapoverlap_omp.inl"
#include "impl/mapoverlap/mapoverlap_cl.inl"
#include "impl/mapoverlap/mapoverlap_cu.inl"
#include "impl/mapoverlap/mapoverlap_hy.inl"

#endif // MAPOVERLAP_H