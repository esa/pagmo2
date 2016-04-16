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

/// Type has fitness() and get_nobj()
/**
 * This type trait defines a static const boolean
 * \p value flag which is \p true if \p T has the following
 * methods implemented:
 *
 * @code
 * fitness_vector fitness(const decision_vector &) const
 * fitness_vector::size_type get_nobj() const 
 * @endcode
 *
 */
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

/// Type has get_bounds()
/**
 * This type trait defines a static const boolean
 * \p value flag which is \p true if \p T has the following
 * method implemented:
 *
 * @code
 * std::pair<vector_double, vector_double> get_bounds() const
 * @endcode
 *
 */
template <typename T>
class has_bounds: detail::sfinae_types
{
        template <typename U>
        static auto test0(const U &p) -> decltype(p.get_bounds());
        static no test0(...);
        static const bool implementation_defined =
            std::is_same<std::pair<vector_double,vector_double>,decltype(test0(std::declval<const T &>()))>::value;
    public:
        static const bool value = implementation_defined;
};

template <typename T>
const bool has_bounds<T>::value;

/// Type has get_nec() and get_nic()
/**
 * This type trait defines a static const boolean
 * \p value flag which is \p true if \p T has the following
 * methods implemented:
 *
 * @code
 * vector_double::size_type get_nec() const
 * vector_double::size_type get_nic() const
 * @endcode
 *
 */
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

/// Type has get_name()
/**
 * This type trait defines a static const boolean
 * \p value flag which is \p true if \p T has the following
 * method implemented:
 *
 * @code
 * std::string get_name() const
 * @endcode
 *
 */
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

/// Type has get_extra_info()
/**
 * This type trait defines a static const boolean
 * \p value flag which is \p true if \p T has the following
 * method implemented:
 *
 * @code
 * std::string get_extra_info() const
 * @endcode
 *
 */
template <typename T>
class has_extra_info: detail::sfinae_types
{
        template <typename U>
        static auto test0(const U &p) -> decltype(p.get_extra_info());
        static no test0(...);
        static const bool implementation_defined =
            std::is_same<std::string,decltype(test0(std::declval<const T &>()))>::value;
    public:
        static const bool value = implementation_defined;
};

template <typename T>
const bool has_extra_info<T>::value;

/// Type has gradient()
/**
 * This type trait defines a static const boolean
 * \p value flag which is \p true if \p T has the following
 * method implemented:
 *
 * @code
 * vector_double gradient(const vector_double &x) const
 * @endcode
 *
 */
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

/// Type has gradient_sparsity()
/**
 * This type trait defines a static const boolean
 * \p value flag which is \p true if \p T has the following
 * method implemented:
 *
 * @code
 * sparsity_pattern gradient_sparsity() const
 * @endcode
 *
 */
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

/// Type has hessians()
/**
 * This type trait defines a static const boolean
 * \p value flag which is \p true if \p T has the following
 * method implemented:
 *
 * @code
 * std::vector<vector_double> hessians(const vector_double &x) const
 * @endcode
 *
 */
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

/// Type has hessians_sparsity()
/**
 * This type trait defines a static const boolean
 * \p value flag which is \p true if \p T has the following
 * method implemented:
 *
 * @code
 * std::vector<sparsity_pattern> hessians_sparsity() const
 * @endcode
 *
 */
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
