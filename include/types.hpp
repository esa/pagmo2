#ifndef PAGMO_TYPES_HPP
#define PAGMO_TYPES_HPP

#include <iostream>
#include <utility>
#include <vector>

namespace pagmo
{

using decision_vector = std::vector<double>;
using fitness_vector = std::vector<double>;
using gradient_vector = std::vector<double>;
using sparsity_pattern = std::vector<std::pair<long,long>>;
using box_bounds = std::pair<decision_vector,decision_vector>;

namespace detail
{
#define PAGMO_MAX_OUTPUT_LENGTH 5u

template <typename ... Args>
void stream(std::ostream &, const Args & ...);

template <typename T>
inline void stream_impl(std::ostream &os, const T &x)
{
    os << x;
}

template <typename T>
inline void stream_impl(std::ostream &os, const std::vector<T> &v)
{
    auto len = v.size();
    if (len < PAGMO_MAX_OUTPUT_LENGTH) 
    {
        os << '[';
        for (decltype(v.size()) i = 0u; i < v.size(); ++i) {
            stream(os, v[i]);
            if (i != v.size() - 1u) {
                os << ", ";
            }
        }
        os << ']';
    } else {
        os << '[';
        for (decltype(v.size()) i = 0u; i < PAGMO_MAX_OUTPUT_LENGTH; ++i) {
            stream(os, v[i], ", ");
        }
        os << " ... ]";
    }
}

template <typename T, typename U>
inline void stream_impl(std::ostream &os, const std::pair<T,U> &p)
{
    stream(os,'(',p.first,',',p.second,')');
}

template <typename T, typename ... Args>
inline void stream_impl(std::ostream &os, const T &x, const Args & ... args)
{
    stream_impl(os,x);
    stream_impl(os,args...);
}

template <typename ... Args>
inline void stream(std::ostream &os, const Args & ... args)
{
    stream_impl(os,args...);
}

template <typename ... Args>
inline void print(const Args & ... args)
{
    stream(std::cout,args...);
}

}} // namespaces

#endif

