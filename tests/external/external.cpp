#include <catch2/catch.hpp>

#include <skepu>


TEST_CASE("External calls the supplied operator")
{
	bool operator_called{false};

	skepu::external([&]
	{
		operator_called = true;
	});

	CHECK(operator_called);
}

struct container_stub
{
	bool flushed;

	auto inline
	flush() noexcept
	-> void
	{
		flushed = true;
	}
};

namespace skepu {

template<>
struct is_skepu_container<container_stub &>
: std::true_type
{};

} // namespace skepu

TEST_CASE("One container is flushed")
{
	container_stub c{false};

	REQUIRE_FALSE(c.flushed);
	REQUIRE_NOTHROW(skepu::external(c, [&]{}));
	CHECK(c.flushed);
}

TEST_CASE("Two containers are flushed")
{
	container_stub c1{false};
	container_stub c2{false};

	REQUIRE_FALSE(c1.flushed);
	REQUIRE_FALSE(c2.flushed);
	REQUIRE_NOTHROW(skepu::external(c1, c2, [&]{}));
	CHECK(c1.flushed);
	CHECK(c2.flushed);
}
