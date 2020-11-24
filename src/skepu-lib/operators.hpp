#include <iostream>

#pragma once

namespace skepu {
namespace lib {


template<typename T>
T add(T lhs, T rhs)
{
  return lhs + rhs;
}

template<typename T>
T sub(T lhs, T rhs)
{
  return lhs - rhs;
}

template<typename T>
T mul(T lhs, T rhs)
{
  return lhs * rhs;
}

template<typename T>
T div(T lhs, T rhs)
{
  return lhs / rhs;
}

}} // skepu::lib
