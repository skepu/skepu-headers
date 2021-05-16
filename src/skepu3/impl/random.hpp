
//#define SKEPU_DEBUG_PRNG
//#define SKEPU_PRNG_VERIFY_FORWARD

namespace skepu
{	
	#define SKEPU_RAND_DEFAULT_SEED 456
	
	#define RND_MOD (1LL << 48)
	#define RND_EXP 1
	#define RND_BASE 0x5deece66d
	#define RND_INC 5
	
	struct RandomForCL
	{
		uint64_t m_state;
	};
	
	struct RandomImpl
	{
		using State = uint64_t;
		using Normalized = double;
		
		State m_state;
#ifdef SKEPU_PRNG_VERIFY_FORWARD
		int m_valid_uses;
#endif
		
		RandomImpl()
		: m_state{0}
#ifdef SKEPU_PRNG_VERIFY_FORWARD
		, m_valid_uses{0}
#endif
		{}
		
		RandomImpl(State state, int uses)
		: m_state{state}
#ifdef SKEPU_PRNG_VERIFY_FORWARD
		, m_valid_uses{uses}
#endif
		{}
		
		State get()
		{
#ifdef SKEPU_PRNG_VERIFY_FORWARD
#pragma omp critical
{
			this->m_valid_uses--;
			if (this->m_valid_uses < 0)
				SKEPU_ERROR("Random object ran out of valid uses");
}
#endif

#ifdef SKEPU_DEBUG_PRNG
			return ++this->m_state;
#else
			for (size_t i = 0; i < RND_EXP; ++i)
				this->m_state = (this->m_state * RND_BASE + RND_INC) % RND_MOD;
			return this->m_state;
#endif
		}
		
		Normalized getNormalized()
		{
			return (Normalized)this->get() / RND_MOD;
		}
		
	};
	
	template<size_t GetCount = 0>
	class Random: public RandomImpl
	{
	public:
		
		Random<GetCount>(State state, int uses)
		: RandomImpl{state, uses}
		{}
		
		Random<GetCount>()
		: RandomImpl{}
		{}
		
		operator Random<>*()
		{
			return reinterpret_cast<Random<>*>(this);
		}
	};
	
	class PRNG
	{
		using State = RandomImpl::State;
		
		State m_state;
		size_t m_next_iterations = 0;
		size_t m_current_iterations = 0;
		void *m_target_instance;
		
	public:
		
		PRNG(State seed = SKEPU_RAND_DEFAULT_SEED)
		: m_state{seed}
		{}
		
		void registerInstance(void *obj, size_t iterations)
		{
			this->m_target_instance = obj;
			this->m_next_iterations = iterations;
		}
		
		void forward(size_t steps)
		{
#ifdef SKEPU_DEBUG_PRNG
			this->m_state += steps;
#else
			for (size_t step = 0; step < steps; ++step)
			{
				for (size_t i = 0; i < RND_EXP; ++i)
					this->m_state = (this->m_state * RND_BASE + RND_INC) % RND_MOD;
			}
#endif
		}
		
		template<size_t GetCount>
		Random<GetCount> asRandom(size_t size)
		{
			size_t uses = size * GetCount;
			Random<GetCount> retval(this->m_state, uses);
			this->forward(uses);
			return retval;
		}
		
		template<size_t GetCount>
		skepu::Vector<Random<GetCount>> asRandom(size_t size, size_t copies, size_t atomic_size = 1)
		{
			// precondition: size | atomic_size
			size_t atomic_chunk_count = size / atomic_size;
			size_t sub_chunk = atomic_chunk_count / copies;
			size_t sub_rest = atomic_chunk_count % copies;
			size_t chunk = sub_chunk * atomic_size;
			size_t rest = sub_rest * atomic_size;
			
			/*
			std::cout << "size: " << size << "\n";
			std::cout << "copies: " << copies << "\n";
			std::cout << "atomic_size: " << atomic_size << "\n";
			std::cout << "atomic_chunk_count: " << atomic_chunk_count << "\n";
			std::cout << "sub_chunk: " << sub_chunk << "\n";
			std::cout << "sub_rest: " << sub_rest << "\n";
			std::cout << "chunk: " << chunk << "\n";
			std::cout << "rest: " << rest << "\n";*/
			
			skepu::Vector<Random<GetCount>> retval_vec(copies);
			for (size_t i = 0; i < copies; ++i)
			{
				size_t uses = (chunk + (i < sub_rest ? atomic_size : 0)) * GetCount;
				retval_vec[i] = Random<GetCount>(this->m_state, uses);
				this->forward(uses);
			}
			
			return retval_vec;
		}
		
		template<size_t GetCount>
		skepu::Vector<RandomForCL> asRandomB(size_t size, size_t copies, size_t atomic_size = 1)
		{
			// precondition: size | atomic_size
			size_t atomic_chunk_count = size / atomic_size;
			size_t sub_chunk = atomic_chunk_count / copies;
			size_t sub_rest = atomic_chunk_count % copies;
			size_t chunk = sub_chunk * atomic_size;
			size_t rest = sub_rest * atomic_size;
			
			skepu::Vector<RandomForCL> retval_vec(copies);
			for (size_t i = 0; i < copies; ++i)
			{
				size_t uses = (chunk + (i < sub_rest ? atomic_size : 0));
				retval_vec[i] = RandomForCL{this->m_state};
				this->forward(uses * GetCount);
			}
			return retval_vec;
		}
		
	};
	
	
	template<typename... Args>
	struct has_random: std::false_type {};

	template<size_t RC, typename... Args>
	struct has_random<skepu::Random<RC> &, Args...>: std::true_type {};
	
	
	template<typename T>
	struct lambda_has_random {};

	template<typename Ret, typename Class, typename... Args>
	struct lambda_has_random<Ret(Class::*)(Args...) const>
	: has_random<Args...> {};

	template<typename Ret, typename Class, typename... Args>
	struct lambda_has_random<Ret(Class::*)(Args...)>
	: has_random<Args...> {};
	
}