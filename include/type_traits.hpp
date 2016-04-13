#ifndef PAGMO_TYPE_TRAITS_HPP
#define PAGMO_TYPE_TRAITS_HPP

#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "types.hpp"

namespace pagmo
{

namespace detail
{

struct sfinae_types
{
    struct yes {};
    struct no {};
};

}

/// Detect fitness availability.
template <typename T>
class has_fitness: detail::sfinae_types
{
        template <typename U>
        static auto test0(const U &p) -> decltype(p.fitness(std::declval<const vector_double &>()));
        static no test0(...);
        template <typename U>
        static auto test1(const U &p) -> decltype(p.get_nobj());
        static no test1(...);
        static const bool implementation_defined =
            std::is_same<vector_double,decltype(test0(std::declval<const T &>()))>::value &&
            std::is_same<vector_double::size_type,decltype(test1(std::declval<const T &>()))>::value;
    public:
        static const bool value = implementation_defined;
};

template <typename T>
const bool has_fitness<T>::value;

/// Detect dimensions and bounds availability.
template <typename T>
class has_dimensions_bounds: detail::sfinae_types
{
        template <typename U>
        static auto test0(const U &p) -> decltype(p.get_n());
        static no test0(...);
        template <typename U>
        static auto test1(const U &p) -> decltype(p.get_bounds());
        static no test1(...);
        static const bool implementation_defined =
            std::is_same<vector_double::size_type,decltype(test0(std::declval<const T &>()))>::value &&
            std::is_same<std::pair<vector_double,vector_double>,decltype(test1(std::declval<const T &>()))>::value;
    public:
        static const bool value = implementation_defined;
};

template <typename T>
const bool has_dimensions_bounds<T>::value;

/// Detect constraints availability.
template <typename T>
class has_constraints: detail::sfinae_types
{
        template <typename U>
        static auto test0(const U &p) -> decltype(p.get_nec());
        static no test0(...);
        template <typename U>
        static auto test1(const U &p) -> decltype(p.get_nic());
        static no test1(...);
        static const bool implementation_defined =
            std::is_same<vector_double::size_type,decltype(test0(std::declval<const T &>()))>::value &&
            std::is_same<vector_double::size_type,decltype(test1(std::declval<const T &>()))>::value;
    public:
        static const bool value = implementation_defined;
};

template <typename T>
const bool has_constraints<T>::value;

/// Detect get_name() availability
template <typename T>
class has_name: detail::sfinae_types
{
        template <typename U>
        static auto test0(const U &p) -> decltype(p.get_name());
        static no test0(...);
        static const bool implementation_defined =
            std::is_same<std::string,decltype(test0(std::declval<const T &>()))>::value;
    public:
        static const bool value = implementation_defined;
};

template <typename T>
const bool has_name<T>::value;

/// Detect extra_info() availability
template <typename T>
class has_extra_info: detail::sfinae_types
{
        template <typename U>
        static auto test0(const U &p) -> decltype(p.extra_info());
        static no test0(...);
        static const bool implementation_defined =
            std::is_same<std::string,decltype(test0(std::declval<const T &>()))>::value;
    public:
        static const bool value = implementation_defined;
};

template <typename T>
const bool has_extra_info<T>::value;

/// Detect gradient() availability
template <typename T>
class has_gradient: detail::sfinae_types
{
        template <typename U>
        static auto test0(const U &p) -> decltype(p.gradient(std::declval<const vector_double &>()));
        static no test0(...);
        static const bool implementation_defined =
            std::is_same<vector_double,decltype(test0(std::declval<const T &>()))>::value;
    public:
        static const bool value = implementation_defined;
};

template <typename T>
const bool has_gradient<T>::value;

/// Detect gradient_sparsity() availability
template <typename T>
class has_gradient_sparsity: detail::sfinae_types
{
        template <typename U>
        static auto test0(const U &p) -> decltype(p.gradient_sparsity());
        static no test0(...);
        static const bool implementation_defined =
            std::is_same<sparsity_pattern,decltype(test0(std::declval<const T &>()))>::value;
    public:
        static const bool value = implementation_defined;
};

template <typename T>
const bool has_gradient_sparsity<T>::value;

/// Detect hessians() availability
template <typename T>
class has_hessians: detail::sfinae_types
{
        template <typename U>
        static auto test0(const U &p) -> decltype(p.hessians(std::declval<const vector_double &>()));
        static no test0(...);
        static const bool implementation_defined =
            std::is_same<std::vector<vector_double>,decltype(test0(std::declval<const T &>()))>::value;
    public:
        static const bool value = implementation_defined;
};

template <typename T>
const bool has_hessians<T>::value;

/// Detect hessians_sparsity() availability
template <typename T>
class has_hessians_sparsity: detail::sfinae_types
{
        template <typename U>
        static auto test0(const U &p) -> decltype(p.hessians_sparsity());
        static no test0(...);
        static const bool implementation_defined =
            std::is_same<std::vector<sparsity_pattern>,decltype(test0(std::declval<const T &>()))>::value;
    public:
        static const bool value = implementation_defined;
};

template <typename T>
const bool has_hessians_sparsity<T>::value;

} // namespace pagmo

#endif
