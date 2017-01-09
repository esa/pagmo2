#ifndef PAGMO_TYPE_TRAITS_HPP
#define PAGMO_TYPE_TRAITS_HPP

#include <cstddef>
#include <initializer_list>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "detail/population_fwd.hpp"
#include "types.hpp"

namespace pagmo
{

inline namespace impl
{

// http://en.cppreference.com/w/cpp/types/void_t
template <typename... Ts>
struct make_void {
    typedef void type;
};

template <typename... Ts>
using void_t = typename make_void<Ts...>::type;

// http://en.cppreference.com/w/cpp/experimental/is_detected
template <class Default, class AlwaysVoid, template <class...> class Op, class... Args>
struct detector {
    using value_t = std::false_type;
    using type = Default;
};

template <class Default, template <class...> class Op, class... Args>
struct detector<Default, void_t<Op<Args...>>, Op, Args...> {
    using value_t = std::true_type;
    using type = Op<Args...>;
};

// http://en.cppreference.com/w/cpp/experimental/nonesuch
struct nonesuch {
    nonesuch() = delete;
    ~nonesuch() = delete;
    nonesuch(nonesuch const &) = delete;
    void operator=(nonesuch const &) = delete;
};

template <template <class...> class Op, class... Args>
using is_detected = typename detector<nonesuch, void, Op, Args...>::value_t;

template <template <class...> class Op, class... Args>
using detected_t = typename detector<nonesuch, void, Op, Args...>::type;

// http://en.cppreference.com/w/cpp/types/conjunction
template <class...>
struct conjunction : std::true_type {
};

template <class B1>
struct conjunction<B1> : B1 {
};

template <class B1, class... Bn>
struct conjunction<B1, Bn...> : std::conditional<B1::value != false, conjunction<Bn...>, B1>::type {
};

// http://en.cppreference.com/w/cpp/types/disjunction
template <class...>
struct disjunction : std::false_type {
};

template <class B1>
struct disjunction<B1> : B1 {
};

template <class B1, class... Bn>
struct disjunction<B1, Bn...> : std::conditional<B1::value != false, B1, disjunction<Bn...>>::type {
};

// http://en.cppreference.com/w/cpp/types/negation
template <class B>
struct negation : std::integral_constant<bool, !B::value> {
};

// std::index_sequence and std::make_index_sequence implementation for C++11. These are available
// in the std library in C++14. Implementation taken from:
// http://stackoverflow.com/questions/17424477/implementation-c14-make-integer-sequence
template <std::size_t... Ints>
struct index_sequence {
    using type = index_sequence;
    using value_type = std::size_t;
    static constexpr std::size_t size() noexcept
    {
        return sizeof...(Ints);
    }
};

template <class Sequence1, class Sequence2>
struct merge_and_renumber;

template <std::size_t... I1, std::size_t... I2>
struct merge_and_renumber<index_sequence<I1...>, index_sequence<I2...>>
    : index_sequence<I1..., (sizeof...(I1) + I2)...> {
};

template <std::size_t N>
struct make_index_sequence
    : merge_and_renumber<typename make_index_sequence<N / 2>::type, typename make_index_sequence<N - N / 2>::type> {
};

template <>
struct make_index_sequence<0> : index_sequence<> {
};

template <>
struct make_index_sequence<1> : index_sequence<0> {
};

template <typename T, typename F, std::size_t... Is>
void apply_to_each_item(T &&t, const F &f, index_sequence<Is...>)
{
    (void)std::initializer_list<int>{0, (void(f(std::get<Is>(std::forward<T>(t)))), 0)...};
}

// Tuple for_each(). Execute the functor f on each element of the input Tuple.
// https://isocpp.org/blog/2015/01/for-each-arg-eric-niebler
// https://www.reddit.com/r/cpp/comments/2tffv3/for_each_argumentsean_parent/
// https://www.reddit.com/r/cpp/comments/33b06v/for_each_in_tuple/
template <class Tuple, class F>
void tuple_for_each(Tuple &&t, const F &f)
{
    apply_to_each_item(std::forward<Tuple>(t), f,
                       make_index_sequence<std::tuple_size<typename std::decay<Tuple>::type>::value>{});
}

// Some handy aliases.
template <typename T>
using uncvref_t = typename std::remove_cv<typename std::remove_reference<T>::type>::type;

template <typename T>
using decay_t = typename std::decay<T>::type;

template <bool B, typename T = void>
using enable_if_t = typename std::enable_if<B, T>::type;
}

namespace detail
{

struct sfinae_types {
    struct yes {
    };
    struct no {
    };
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
class has_fitness : detail::sfinae_types
{
    template <typename U>
    static auto test0(const U &p) -> decltype(p.fitness(std::declval<const vector_double &>()));
    static no test0(...);
    static const bool implementation_defined
        = std::is_same<vector_double, decltype(test0(std::declval<const T &>()))>::value;

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
class has_get_nobj : detail::sfinae_types
{
    template <typename U>
    static auto test0(const U &p) -> decltype(p.get_nobj());
    static no test0(...);
    static const bool implementation_defined
        = std::is_same<vector_double::size_type, decltype(test0(std::declval<const T &>()))>::value;

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
class has_bounds : detail::sfinae_types
{
    template <typename U>
    static auto test0(const U &p) -> decltype(p.get_bounds());
    static no test0(...);
    static const bool implementation_defined
        = std::is_same<std::pair<vector_double, vector_double>, decltype(test0(std::declval<const T &>()))>::value;

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
class has_e_constraints : detail::sfinae_types
{
    template <typename U>
    static auto test0(const U &p) -> decltype(p.get_nec());
    static no test0(...);
    static const bool implementation_defined
        = std::is_same<vector_double::size_type, decltype(test0(std::declval<const T &>()))>::value;

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
class has_i_constraints : detail::sfinae_types
{
    template <typename U>
    static auto test0(const U &p) -> decltype(p.get_nic());
    static no test0(...);
    static const bool implementation_defined
        = std::is_same<vector_double::size_type, decltype(test0(std::declval<const T &>()))>::value;

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
class has_set_seed : detail::sfinae_types
{
    template <typename U>
    static auto test0(U &p) -> decltype(p.set_seed(std::declval<unsigned int>()));
    static no test0(...);
    static const bool implementation_defined = std::is_same<void, decltype(test0(std::declval<T &>()))>::value;

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
class override_has_set_seed : detail::sfinae_types
{
    template <typename U>
    static auto test0(const U &p) -> decltype(p.has_set_seed());
    static no test0(...);
    static const bool implementation_defined = std::is_same<bool, decltype(test0(std::declval<const T &>()))>::value;

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
class has_name : detail::sfinae_types
{
    template <typename U>
    static auto test0(const U &p) -> decltype(p.get_name());
    static no test0(...);
    static const bool implementation_defined
        = std::is_same<std::string, decltype(test0(std::declval<const T &>()))>::value;

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
class has_extra_info : detail::sfinae_types
{
    template <typename U>
    static auto test0(const U &p) -> decltype(p.get_extra_info());
    static no test0(...);
    static const bool implementation_defined
        = std::is_same<std::string, decltype(test0(std::declval<const T &>()))>::value;

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
class has_gradient : detail::sfinae_types
{
    template <typename U>
    static auto test0(const U &p) -> decltype(p.gradient(std::declval<const vector_double &>()));
    static no test0(...);
    static const bool implementation_defined
        = std::is_same<vector_double, decltype(test0(std::declval<const T &>()))>::value;

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
class override_has_gradient : detail::sfinae_types
{
    template <typename U>
    static auto test0(const U &p) -> decltype(p.has_gradient());
    static no test0(...);
    static const bool implementation_defined = std::is_same<bool, decltype(test0(std::declval<const T &>()))>::value;

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
class has_gradient_sparsity : detail::sfinae_types
{
    template <typename U>
    static auto test0(const U &p) -> decltype(p.gradient_sparsity());
    static no test0(...);
    static const bool implementation_defined
        = std::is_same<sparsity_pattern, decltype(test0(std::declval<const T &>()))>::value;

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
class has_hessians : detail::sfinae_types
{
    template <typename U>
    static auto test0(const U &p) -> decltype(p.hessians(std::declval<const vector_double &>()));
    static no test0(...);
    static const bool implementation_defined
        = std::is_same<std::vector<vector_double>, decltype(test0(std::declval<const T &>()))>::value;

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
class override_has_hessians : detail::sfinae_types
{
    template <typename U>
    static auto test0(const U &p) -> decltype(p.has_hessians());
    static no test0(...);
    static const bool implementation_defined = std::is_same<bool, decltype(test0(std::declval<const T &>()))>::value;

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
class has_hessians_sparsity : detail::sfinae_types
{
    template <typename U>
    static auto test0(const U &p) -> decltype(p.hessians_sparsity());
    static no test0(...);
    static const bool implementation_defined
        = std::is_same<std::vector<sparsity_pattern>, decltype(test0(std::declval<const T &>()))>::value;

public:
    /// static const boolean value flag
    static const bool value = implementation_defined;
};

template <typename T>
const bool has_hessians_sparsity<T>::value;

/// Type has has_gradient_sparsity()
/**
 * This type trait defines a static const boolean
 * \p value flag which is \p true if \p T has the following
 * method implemented:
 *
 * @code
 * bool has_gradient_sparsity() const
 * @endcode
 */
template <typename T>
class override_has_gradient_sparsity : detail::sfinae_types
{
    template <typename U>
    static auto test0(const U &p) -> decltype(p.has_gradient_sparsity());
    static no test0(...);
    static const bool implementation_defined = std::is_same<bool, decltype(test0(std::declval<const T &>()))>::value;

public:
    /// static const boolean value flag
    static const bool value = implementation_defined;
};

template <typename T>
const bool override_has_gradient_sparsity<T>::value;

/// Type has has_hessians_sparsity()
/**
 * This type trait defines a static const boolean
 * \p value flag which is \p true if \p T has the following
 * method implemented:
 *
 * @code
 * bool has_hessians_sparsity() const
 * @endcode
 */
template <typename T>
class override_has_hessians_sparsity : detail::sfinae_types
{
    template <typename U>
    static auto test0(const U &p) -> decltype(p.has_hessians_sparsity());
    static no test0(...);
    static const bool implementation_defined = std::is_same<bool, decltype(test0(std::declval<const T &>()))>::value;

public:
    /// static const boolean value flag
    static const bool value = implementation_defined;
};

template <typename T>
const bool override_has_hessians_sparsity<T>::value;

} // namespace pagmo

#endif
