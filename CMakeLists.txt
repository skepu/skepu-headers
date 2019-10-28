cmake_minimum_required(VERSION 3.10)

if(PROJECT_NAME)
	set(skepu_headers_SUBPROJECT ON)
endif()

project(skepu-headers
	VERSION 3.0.0
	LANGUAGES CXX)
include(CMakePackageConfigHelpers)

add_library(skepu-headers INTERFACE)
# TODO:
# * Should skepu-headers have a namespace?
target_compile_features(skepu-headers INTERFACE
	cxx_attributes
	cxx_auto_type
	cxx_constexpr
	cxx_decltype
	cxx_lambdas
	cxx_range_for
	cxx_rvalue_references
	cxx_variadic_templates
	cxx_template_template_parameters)
target_include_directories(skepu-headers INTERFACE
	$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src>
	$<INSTALL_INTERFACE:include>)

if(NOT skepu_headers_SUBPROJECT)
	# Configure some basic stuff that CMake does not do right...
	if(NOT CMAKE_BUILD_TYPE)
		set(CMAKE_BUILD_TYPE "Debug")
	else()
		if(NOT (CMAKE_BUILD_TYPE STREQUAL "Debug"
						OR CMAKE_BUILD_TYPE STREQUAL "Release"))
			message(FATAL_ERROR "The build type must be either Debug or Release")
		endif()
	endif()
	set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE
		STRING "Debug or Release build." FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
		"Debug" "Release")

	if(NOT DEFINED ENABLE_TESTING)
		if(CMAKE_BUILD_TYPE STREQUAL "Debug")
			set(ENABLE_TESTING ON)
		else()
			set(ENABLE_TESTING OFF)
		endif()
	endif()
	set(ENABLE_TESTING ${ENABLE_TESTING} CACHE BOOL
		"Enables the test suite." FORCE)
	if(ENABLE_TESTING)
		enable_testing()
		add_subdirectory(tests)
	else()
		file(REMOVE ${CMAKE_BINARY_DIR}/CTestTestfile.cmake)
	endif()
endif()

option(skepu_headers_INSTALL
	"Install SkePU headers even if subproject."
	OFF)

if(NOT skepu_headers_SUBPROJECT
		OR skepu_headers_INSTALL)
	install(TARGETS skepu-headers
		EXPORT skepu-headersTargets
		DESTINATION lib)
	install(EXPORT skepu-headersTargets
		#NAMESPACE skepu-headers::
		DESTINATION lib/cmake/skepu-headers)

	configure_package_config_file(
		${CMAKE_CURRENT_LIST_DIR}/CMake/skepu-headersConfig.cmake.in
		${CMAKE_CURRENT_BINARY_DIR}/skepu-headersConfig.cmake
		INSTALL_DESTINATION lib/cmake/skepu-headers)
	write_basic_package_version_file(
		${CMAKE_CURRENT_BINARY_DIR}/skepu-headersConfigVersion.cmake
		COMPATIBILITY SameMajorVersion)
	install(
		FILES
			${CMAKE_CURRENT_BINARY_DIR}/skepu-headersConfig.cmake
			${CMAKE_CURRENT_BINARY_DIR}/skepu-headersConfigVersion.cmake
		DESTINATION lib/cmake/skepu-headers)

	install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/src/skepu3
		DESTINATION include)
	install(FILES ${CMAKE_CURRENT_LIST_DIR}/src/skepu
		DESTINATION include)

# TODO:
# * SkePU API Reference
endif()

# vi:ts=2 sw=2 noet
