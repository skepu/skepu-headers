#include <iostream>

#pragma once

namespace skepu {
namespace experimental {
namespace complex {


// Cannot have type templates in SkePU (yet)
struct DComplex
{
    double re;
    double im;
};

struct FComplex
{
    float re;
    float im;
};


// Type traits

template<typename T>
struct value_type {};

template<>
struct value_type<DComplex> { using type = double; };

template<>
struct value_type<FComplex> { using type = float; };


template<typename T>
struct is_skepu_complex: std::false_type {};

template<>
struct is_skepu_complex<DComplex>: std::true_type {};

template<>
struct is_skepu_complex<FComplex>: std::true_type {};


// Constants

// Cannot be used inside userfunctions, usable e.g. for default values to Reduce skeleton invocations
constexpr DComplex DZero = {0.0, 0.0};
constexpr DComplex DOne  = {1.0, 0.0};

constexpr FComplex FZero = {0.0, 0.0};
constexpr FComplex FOne = {1.0, 0.0};


// Functions

template<typename C>
inline
C add(C lhs, C rhs)
{
  C res;
  res.re = lhs.re + rhs.re;
  res.im = lhs.im + rhs.im;
  return res;
}

template<typename C>
inline
C sub(C lhs, C rhs)
{
  C res;
  res.re = lhs.re - rhs.re;
  res.im = lhs.im - rhs.im;
  return res;
}

template<typename C>
inline
C mul(C lhs, C rhs)
{
  C res;
  res.re = lhs.re * rhs.re - lhs.im * rhs.im;
  res.im = lhs.im * rhs.re + lhs.re * rhs.im;
  return res;
}

template<typename C>
inline
C real_div(C z, typename value_type<C>::type div)
{
  C res;
  res.re = z.re / div;
  res.im = z.im / div;
  return res;
}

template<typename C>
inline
C conj(C z)
{
  C res;
  res.re = z.re;
  res.im = -z.im;
  return res;
}

template<typename C>
inline
typename value_type<C>::type sq_norm(C z)
{
  return z.re * z.re + z.im * z.im;
}




// For use outside skeletons

template<typename C>
inline 
typename std::enable_if<is_skepu_complex<C>::value, std::ostream &>::type
operator<<(std::ostream &o, C z)
{
  o << z.re << (z.im >= 0 ? "+" : "") << z.im << "i";
  return o;
}

}}} // skepu::experimental::complex
