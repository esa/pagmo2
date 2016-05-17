#ifndef PAGMO_TYPE_TRAITS_HPP
#define PAGMO_TYPE_TRAITS_HPP

#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "detail/population_fwd.hpp"
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

/// Type has fitness()
/**
 * This type trait defines a static const boolean
 * \p value flag which is \p true if \p T has the following
 * methods implemented:
 *
 * @code
 * fitness_vector fitness(const decision_vector &) const
 * @endcode
 *
 */
template <typename T>
class has_fitness: detail::sfinae_types
{
        template <typename U>
        static auto test0(const U &p) -> decltype(p.fitness(std::declval<const vector_double &>()));
        static no test0(...);
        static const bool implementation_defined =
            std::is_same<vector_double,decltype(test0(std::declval<const T &>()))>::value;
    public:
        /// static const boolean value flag
        static const bool value = implementation_defined;
};

template <typename T>
const bool has_fitness<T>::value;

/// Type has get_nobj()
/**
 * This type trait defines a static const boolean
 * \p value flag which is \p true if \p T has the following
 * methods implemented:
 *
 * @code
 * fitness_vector::size_type get_nobj() const
 * @endcode
 *
 */
template <typename T>
class has_get_nobj: detail::sfinae_types
{
        template <typename U>
        static auto test0(const U &p) -> decltype(p.get_nobj());
        static no test0(...);
        static const bool implementation_defined =
            std::is_same<vector_double::size_type,decltype(test0(std::declval<const T &>()))>::value;
    public:
        /// static const boolean value flag
        static const bool value = implementation_defined;
};

template <typename T>
const bool has_get_nobj<T>::value;

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
        /// static const boolean value flag
        static const bool value = implementation_defined;
};

template <typename T>
const bool has_bounds<T>::value;

/// Type has get_nec()
/**
 * This type trait defines a static const boolean
 * \p value flag which is \p true if \p T has the following
 * method implemented:
 *
 * @code
 * vector_double::size_type get_nec() const
 * @endcode
 *
 */
template <typename T>
class has_e_constraints: detail::sfinae_types
{
        template <typename U>
        static auto test0(const U &p) -> decltype(p.get_nec());
        static no test0(...);
        static const bool implementation_defined =
            std::is_same<vector_double::size_type,decltype(test0(std::declval<const T &>()))>::value;
    public:
        /// static const boolean value flag
        static const bool value = implementation_defined;
};

template <typename T>
const bool has_e_constraints<T>::value;

/// Type has get_nic()
/**
 * This type trait defines a static const boolean
 * \p value flag which is \p true if \p T has the following
 * method implemented:
 *
 * @code
 * vector_double::size_type get_nic() const
 * @endcode
 *
 */
template <typename T>
class has_i_constraints: detail::sfinae_types
{
        template <typename U>
        static auto test0(const U &p) -> decltype(p.get_nic());
        static no test0(...);
        static const bool implementation_defined =
            std::is_same<vector_double::size_type,decltype(test0(std::declval<const T &>()))>::value;
    public:
        /// static const boolean value flag
        static const bool value = implementation_defined;
};

template <typename T>
const bool has_i_constraints<T>::value;

/// Type has set_seed()
/**
 * This type trait defines a static const boolean
 * \p value flag which is \p true if the expression p.set_seed(n)
 * is valid and returns void, where p is a non-const instance of \p T and n is an unsigned int
 *
 * For example, if \p T has the following method implemented:
 *
 * @code
 * void set_seed(unsigned int seed)
 * @endcode
 *
 */
template <typename T>
class has_set_seed: detail::sfinae_types
{
        template <typename U>
        static auto test0(U &p) -> decltype(p.set_seed(std::declval<unsigned int>()));
        static no test0(...);
        static const bool implementation_defined =
            std::is_same<void,decltype(test0(std::declval<T &>()))>::value;
    public:
        /// static const boolean value flag
        static const bool value = implementation_defined;
};

template <typename T>
const bool has_set_seed<T>::value;

/// Type has has_set_seed()
/**
 * This type trait defines a static const boolean
 * \p value flag which is \p true if \p T has the following
 * method implemented:
 *
 * @code
 * bool has_set_seed() const
 * @endcode
 *
 */
template <typename T>
class override_has_set_seed: detail::sfinae_types
{
        template <typename U>
        static auto test0(const U &p) -> decltype(p.has_set_seed());
        static no test0(...);
        static const bool implementation_defined =
            std::is_same<bool,decltype(test0(std::declval<const T &>()))>::value;
    public:
        /// static const boolean value flag
        static const bool value = implementation_defined;
};

template <typename T>
const bool override_has_set_seed<T>::value;

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
        /// static const boolean value flag
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
        /// static const boolean value flag
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
        /// static const boolean value flag
        static const bool value = implementation_defined;
};

template <typename T>
const bool has_gradient<T>::value;


/// Type has has_gradient()
/**
 * This type trait defines a static const boolean
 * \p value flag which is \p true if \p T has the following
 * method implemented:
 *
 * @code
 * bool has_gradient() const
 * @endcode
 *
 */
template <typename T>
class override_has_gradient: detail::sfinae_types
{
        template <typename U>
        static auto test0(const U &p) -> decltype(p.has_gradient());
        static no test0(...);
        static const bool implementation_defined =
            std::is_same<bool,decltype(test0(std::declval<const T &>()))>::value;
    public:
        /// static const boolean value flag
        static const bool value = implementation_defined;
};

template <typename T>
const bool override_has_gradient<T>::value;

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
        /// static const boolean value flag
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
        /// static const boolean value flag
        static const bool value = implementation_defined;
};

template <typename T>
const bool has_hessians<T>::value;

/// Type has has_hessians()
/**
 * This type trait defines a static const boolean
 * \p value flag which is \p true if \p T has the following
 * method implemented:
 *
 * @code
 * bool has_hessians() const
 * @endcode
 *
 */
template <typename T>
class override_has_hessians: detail::sfinae_types
{
        template <typename U>
        static auto test0(const U &p) -> decltype(p.has_hessians());
        static no test0(...);
        static const bool implementation_defined =
            std::is_same<bool,decltype(test0(std::declval<const T &>()))>::value;
    public:
        /// static const boolean value flag
        static const bool value = implementation_defined;
};

template <typename T>
const bool override_has_hessians<T>::value;

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
        /// static const boolean value flag
        static const bool value = implementation_defined;
};

template <typename T>
const bool has_hessians_sparsity<T>::value;

/// Type has set_verbose
/**
 * This type trait defines a static const boolean
 * \p value flag which is \p true if the expression p.set_verbose(n)
 * is valid and returns void, where p is a non-const instance of \p T and n is a bool
 *
 * For example, if \p T has the following method implemented:
 *
 * @code
 * void set_verbose(unsigned int level)
 * @endcode
 *
 */
template <typename T>
class has_set_verbosity: detail::sfinae_types
{
        template <typename U>
        static auto test0(U &p) -> decltype(p.set_verbosity(std::declval<unsigned int>()));
        static no test0(...);
        static const bool implementation_defined =
            std::is_same<void,decltype(test0(std::declval<T &>()))>::value;
    public:
        /// static const boolean value flag
        static const bool value = implementation_defined;
};

template <typename T>
const bool has_set_verbosity<T>::value;

/// Type has has_set_verbosity()
/**
 * This type trait defines a static const boolean
 * \p value flag which is \p true if \p T has the following
 * method implemented:
 *
 * @code
 * bool has_set_verbosity() const
 * @endcode
 *
 */
template <typename T>
class override_has_set_verbosity: detail::sfinae_types
{
        template <typename U>
        static auto test0(const U &p) -> decltype(p.has_set_verbosity());
        static no test0(...);
        static const bool implementation_defined =
            std::is_same<bool,decltype(test0(std::declval<const T &>()))>::value;
    public:
        /// static const boolean value flag
        static const bool value = implementation_defined;
};

template <typename T>
const bool override_has_set_verbosity<T>::value;

/// Type has evolve
/**
 * This type trait defines a static const boolean
 * \p value flag which is \p true if \p T has the following
 * method implemented:
 *
 * @code
 * population evolve(population pop) const
 * @endcode
 */
template <typename T>
class has_evolve: detail::sfinae_types
{
        template <typename U>
        static auto test0(const U &p) -> decltype(p.evolve(std::declval<population>()));
        static no test0(...);
        static const bool implementation_defined =
            std::is_same<population,decltype(test0(std::declval<const T &>()))>::value;
    public:
        /// static const boolean value flag
        static const bool value = implementation_defined;
};

template <typename T>
const bool has_evolve<T>::value;

} // namespace pagmo

#endif
