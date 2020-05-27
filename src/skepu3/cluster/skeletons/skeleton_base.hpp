#pragma once
#ifndef SKEPU_CLUSTER_SKELETON_BASE_HPP
#define SKEPU_CLUSTER_SKELETON_BASE_HPP 1

#include <skepu3/cluster/common.hpp>

namespace skepu {

namespace backend {

class SkeletonBase
{
public:
	virtual ~SkeletonBase() noexcept {};

	auto finishAll() noexcept -> void {};
	auto setExecPlan(ExecPlan) noexcept -> void {};
	auto setBackend(BackendSpec) noexcept -> void {};
	auto resetBackend() noexcept -> void {};

protected:
	SkeletonBase() noexcept {}
}; // class SkeletonBase

} // namespace backend

} // namespace skepu


#endif // SKEPU_CLUSTER_SKELETON_BASE_HPP
