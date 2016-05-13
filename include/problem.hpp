#ifndef PAGMO_PROBLEM_HPP
#define PAGMO_PROBLEM_HPP

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>

#include "exceptions.hpp"
#include "io.hpp"
#include "serialization.hpp"
#include "type_traits.hpp"

/// Macro for the registration of the serialization functionality for problems.
/**
 * This macro should always be invoked after the declaration of a problem: it will register
 * the problem with PaGMO's serialization machinery. The macro should be called in the root namespace
 * and using the fully qualified name of the problem to be registered. For example:
 * @code
 * namespace my_namespace
 * {
 *
 * class my_problem
 * {
 *    // ...
 * };
 *
 * }
 *
 * PAGMO_REGISTER_PROBLEM(my_namespace::my_problem)
 * @endcode
 */
#define PAGMO_REGISTER_PROBLEM(prob) CEREAL_REGISTER_TYPE_WITH_NAME(pagmo::detail::prob_inner<prob>,#prob)

namespace pagmo
{

namespace detail
{

// Helper to check that the problem bounds are valid. This will throw if the bounds
// are invalid because of:
// - the bounds size is zero,
// - inconsistent lengths of the vectors,
// - nans in the bounds,
// - lower bounds greater than upper bounds.
inline void check_problem_bounds(const std::pair<vector_double,vector_double> &bounds)
{
    const auto &lb = bounds.first;
    const auto &ub = bounds.second;
    // 0 - Check that the size is at least 1.
    if (lb.size() == 0u) {
        pagmo_throw(std::invalid_argument,"The bounds dimension cannot be zero");
    }
    // 1 - check bounds have equal length
    if (lb.size()!=ub.size()) {
        pagmo_throw(std::invalid_argument,"Length of lower bounds vector is " + std::to_string(lb.size()) +
            ", length of upper bound is " + std::to_string(ub.size()));
    }
    // 2 - checks lower < upper for all values in lb, ub, and check for nans.
    for (decltype(lb.size()) i=0u; i < lb.size(); ++i) {
        if (std::isnan(lb[i]) || std::isnan(ub[i])) {
            pagmo_throw(std::invalid_argument,"A NaN value was encountered in the problem bounds, index: " + std::to_string(i) );
        }
        if (lb[i] > ub[i]) {
            pagmo_throw(std::invalid_argument,"The lower bound at position " + std::to_string(i) + " is " + std::to_string(lb[i]) +
                " while the upper bound has the smaller value " + std::to_string(ub[i]));
        }
    }
}

struct prob_inner_base
{
    virtual ~prob_inner_base() {}
    virtual prob_inner_base *clone() const = 0;
    virtual vector_double fitness(const vector_double &) const = 0;
    virtual vector_double gradient(const vector_double &) const = 0;
    virtual bool has_gradient() const = 0;
    virtual sparsity_pattern gradient_sparsity() const = 0;
    virtual bool has_gradient_sparsity() const = 0;
    virtual std::vector<vector_double> hessians(const vector_double &) const = 0;
    virtual bool has_hessians() const = 0;
    virtual std::vector<sparsity_pattern> hessians_sparsity() const = 0;
    virtual bool has_hessians_sparsity() const = 0;
    virtual vector_double::size_type get_nobj() const = 0;
    virtual std::pair<vector_double,vector_double> get_bounds() const = 0;
    virtual vector_double::size_type get_nec() const = 0;
    virtual vector_double::size_type get_nic() const = 0;
    virtual void set_seed(unsigned int) = 0;
    virtual bool has_set_seed() const = 0;
    virtual std::string get_name() const = 0;
    virtual std::string get_extra_info() const = 0;
    template <typename Archive>
    void serialize(Archive &) {}
};

template <typename T>
struct prob_inner final: prob_inner_base
{
    // Static checks.
    static_assert(
        std::is_default_constructible<T>::value &&
        std::is_copy_constructible<T>::value &&
        std::is_move_constructible<T>::value &&
        std::is_destructible<T>::value,
        "A problem must be default-constructible, copy-constructible, move-constructible and destructible."
    );
    static_assert(has_fitness<T>::value,
        "A problem must provide a fitness function 'vector_double fitness(const vector_double &x) const' and "
        "a method to query the number of objectives 'vector_double::size_type get_nobj() const'.");
    static_assert(has_bounds<T>::value,
        "A problem must provide getters for its bounds [std::pair<vector_double, vector_double> get_bounds() const].");
    // We just need the def ctor, delete everything else.
    prob_inner() = default;
    prob_inner(const prob_inner &) = delete;
    prob_inner(prob_inner &&) = delete;
    prob_inner &operator=(const prob_inner &) = delete;
    prob_inner &operator=(prob_inner &&) = delete;
    // Constructors from T (copy and move variants).
    explicit prob_inner(const T &x):m_value(x) {}
    explicit prob_inner(T &&x):m_value(std::move(x)) {}
    // The clone method, used in the copy constructor of problem.
    virtual prob_inner_base *clone() const override final
    {
        return ::new prob_inner(m_value);
    }
    // Mandatory methods.
    virtual vector_double fitness(const vector_double &dv) const override final
    {
        return m_value.fitness(dv);
    }
    virtual vector_double::size_type get_nobj() const override final
    {
        return m_value.get_nobj();
    }
    virtual std::pair<vector_double,vector_double> get_bounds() const override final
    {
        return m_value.get_bounds();
    }
    // optional methods
    virtual vector_double gradient(const vector_double &dv) const override final
    {
        return gradient_impl(m_value, dv);
    }
    virtual bool has_gradient() const override final
    {
        return has_gradient_impl(m_value);
    }
    virtual sparsity_pattern gradient_sparsity() const override final
    {
        return gradient_sparsity_impl(m_value);
    }
    virtual bool has_gradient_sparsity() const override final
    {
        return has_gradient_sparsity_impl(m_value);
    }
    virtual std::vector<vector_double> hessians(const vector_double &dv) const override final
    {
        return hessians_impl(m_value, dv);
    }
    virtual bool has_hessians() const override final
    {
        return has_hessians_impl(m_value);
    }
    virtual std::vector<sparsity_pattern> hessians_sparsity() const override final
    {
        return hessians_sparsity_impl(m_value);
    }
    virtual bool has_hessians_sparsity() const override final
    {
        return has_hessians_sparsity_impl(m_value);
    }
    virtual vector_double::size_type get_nec() const override final
    {
        return get_nec_impl(m_value);
    }
    virtual vector_double::size_type get_nic() const override final
    {
        return get_nic_impl(m_value);
    }
    virtual void set_seed(unsigned int seed) override final
    {
        set_seed_impl(m_value, seed);
    }
    virtual bool has_set_seed() const override final
    {
        return has_set_seed_impl(m_value);
    }
    virtual std::string get_name() const override final
    {
        return get_name_impl(m_value);
    }
    virtual std::string get_extra_info() const override final
    {
        return get_extra_info_impl(m_value);
    }
    // Implementation of the optional methods.
    template <typename U, typename std::enable_if<pagmo::has_gradient<U>::value,int>::type = 0>
    static vector_double gradient_impl(const U &value, const vector_double &dv)
    {
        return value.gradient(dv);
    }
    template <typename U, typename std::enable_if<!pagmo::has_gradient<U>::value,int>::type = 0>
    static vector_double gradient_impl(const U &, const vector_double &)
    {
        pagmo_throw(std::logic_error,"Gradients have been requested but not implemented.\n"
            "A function with prototype 'vector_double gradient(const vector_double &x)' const was expected.");
    }
    template <typename U, typename std::enable_if<pagmo::has_gradient<U>::value && pagmo::override_has_gradient<U>::value,int>::type = 0>
    static bool has_gradient_impl(const U &p)
    {
       return p.has_gradient();
    }
    template <typename U, typename std::enable_if<pagmo::has_gradient<U>::value && !pagmo::override_has_gradient<U>::value,int>::type = 0>
    static bool has_gradient_impl(const U &)
    {
       return true;
    }
    template <typename U, typename std::enable_if<!pagmo::has_gradient<U>::value,int>::type = 0>
    static bool has_gradient_impl(const U &)
    {
       return false;
    }
    template <typename U, typename std::enable_if<pagmo::has_gradient_sparsity<U>::value,int>::type = 0>
    sparsity_pattern gradient_sparsity_impl(const U &p) const
    {
        return p.gradient_sparsity();
    }
    template <typename U, typename std::enable_if<!pagmo::has_gradient_sparsity<U>::value,int>::type = 0>
    sparsity_pattern gradient_sparsity_impl(const U &) const
    {
        pagmo_throw(std::logic_error,"Trying to access non-existing 'gradient_sparsity()' method. This "
            "indicates a logical error in the implementation of the concrete problem class, as the 'gradient_sparsity()' "
            "method is accessed only if 'has_gradient_sparsity()' returns true.");
    }
    template <typename U, typename std::enable_if<pagmo::has_gradient_sparsity<U>::value && pagmo::override_has_gradient_sparsity<U>::value,int>::type = 0>
    static bool has_gradient_sparsity_impl(const U &p)
    {
       return p.has_gradient_sparsity();
    }
    template <typename U, typename std::enable_if<pagmo::has_gradient_sparsity<U>::value && !pagmo::override_has_gradient_sparsity<U>::value,int>::type = 0>
    static bool has_gradient_sparsity_impl(const U &)
    {
       return true;
    }
    template <typename U, typename std::enable_if<!pagmo::has_gradient_sparsity<U>::value,int>::type = 0>
    static bool has_gradient_sparsity_impl(const U &)
    {
       return false;
    }
    template <typename U, typename std::enable_if<pagmo::has_hessians<U>::value,int>::type = 0>
    static std::vector<vector_double> hessians_impl(const U &value, const vector_double &dv)
    {
        return value.hessians(dv);
    }
    template <typename U, typename std::enable_if<!pagmo::has_hessians<U>::value,int>::type = 0>
    static std::vector<vector_double> hessians_impl(const U &, const vector_double &)
    {
        pagmo_throw(std::logic_error,"Hessians have been requested but not implemented.\n"
            "A function with prototype 'std::vector<vector_double> hessians(const vector_double &x)' const was expected.");
    }
    template <typename U, typename std::enable_if<pagmo::has_hessians<U>::value && pagmo::override_has_hessians<U>::value,int>::type = 0>
    static bool has_hessians_impl(const U &p)
    {
       return p.has_hessians();
    }
    template <typename U, typename std::enable_if<pagmo::has_hessians<U>::value && !pagmo::override_has_hessians<U>::value,int>::type = 0>
    static bool has_hessians_impl(const U &)
    {
       return true;
    }
    template <typename U, typename std::enable_if<!pagmo::has_hessians<U>::value,int>::type = 0>
    static bool has_hessians_impl(const U &)
    {
       return false;
    }
    template <typename U, typename std::enable_if<pagmo::has_hessians_sparsity<U>::value,int>::type = 0>
    std::vector<sparsity_pattern> hessians_sparsity_impl(const U &value) const
    {
        return value.hessians_sparsity();
    }
    template <typename U, typename std::enable_if<!pagmo::has_hessians_sparsity<U>::value,int>::type = 0>
    std::vector<sparsity_pattern> hessians_sparsity_impl(const U &) const
    {
        pagmo_throw(std::logic_error,"Trying to access non-existing 'hessians_sparsity()' method. This "
            "indicates a logical error in the implementation of the concrete problem class, as the 'hessians_sparsity()' "
            "method is accessed only if 'has_hessians_sparsity()' returns true.");
    }
    template <typename U, typename std::enable_if<pagmo::has_hessians_sparsity<U>::value && pagmo::override_has_hessians_sparsity<U>::value,int>::type = 0>
    static bool has_hessians_sparsity_impl(const U &p)
    {
       return p.has_hessians_sparsity();
    }
    template <typename U, typename std::enable_if<pagmo::has_hessians_sparsity<U>::value && !pagmo::override_has_hessians_sparsity<U>::value,int>::type = 0>
    static bool has_hessians_sparsity_impl(const U &)
    {
       return true;
    }
    template <typename U, typename std::enable_if<!pagmo::has_hessians_sparsity<U>::value,int>::type = 0>
    static bool has_hessians_sparsity_impl(const U &)
    {
       return false;
    }
    template <typename U, typename std::enable_if<has_e_constraints<U>::value,int>::type = 0>
    static vector_double::size_type get_nec_impl(const U &value)
    {
        return value.get_nec();
    }
    template <typename U, typename std::enable_if<!has_e_constraints<U>::value,int>::type = 0>
    static vector_double::size_type get_nec_impl(const U &)
    {
        return 0;
    }
    template <typename U, typename std::enable_if<has_i_constraints<U>::value,int>::type = 0>
    static vector_double::size_type get_nic_impl(const U &value)
    {
        return value.get_nic();
    }
    template <typename U, typename std::enable_if<!has_i_constraints<U>::value,int>::type = 0>
    static vector_double::size_type get_nic_impl(const U &)
    {
        return 0;
    }
    template <typename U, typename std::enable_if<pagmo::has_set_seed<U>::value,int>::type = 0>
    static void set_seed_impl(U &value, unsigned int seed)
    {
        value.set_seed(seed);
    }
    template <typename U, typename std::enable_if<!pagmo::has_set_seed<U>::value,int>::type = 0>
    static void set_seed_impl(U &, unsigned int)
    {
        pagmo_throw(std::logic_error,"The set_seed method has been called but not implemented by the user.\n"
            "A function with prototype 'void set_seed(unsigned int)' was expected in the user defined problem.");
    }
    template <typename U, typename std::enable_if<pagmo::has_set_seed<U>::value && override_has_set_seed<U>::value,int>::type = 0>
    static bool has_set_seed_impl(const U &p)
    {
       return p.has_set_seed();
    }
    template <typename U, typename std::enable_if<pagmo::has_set_seed<U>::value && !override_has_set_seed<U>::value,int>::type = 0>
    static bool has_set_seed_impl(const U &)
    {
       return true;
    }
    template <typename U, typename std::enable_if<!pagmo::has_set_seed<U>::value,int>::type = 0>
    static bool has_set_seed_impl(const U &)
    {
       return false;
    }
    template <typename U, typename std::enable_if<has_name<U>::value,int>::type = 0>
    static std::string get_name_impl(const U &value)
    {
        return value.get_name();
    }
    template <typename U, typename std::enable_if<!has_name<U>::value,int>::type = 0>
    static std::string get_name_impl(const U &)
    {
        return typeid(U).name();
    }
    template <typename U, typename std::enable_if<has_extra_info<U>::value,int>::type = 0>
    static std::string get_extra_info_impl(const U &value)
    {
        return value.get_extra_info();
    }
    template <typename U, typename std::enable_if<!has_extra_info<U>::value,int>::type = 0>
    static std::string get_extra_info_impl(const U &)
    {
        return "";
    }

    // Serialization.
    template <typename Archive>
    void serialize(Archive &ar)
    {
        ar(cereal::base_class<prob_inner_base>(this),m_value);
    }
    T m_value;
};

} // namespace detail

/// Problem class.
/**
 * This class represents a generic *mathematical programming* or *evolutionary optimization* problem in the form:
 * \f[
 * \begin{array}{rl}
 * \mbox{find:}      & \mathbf {lb} \le \mathbf x \le \mathbf{ub}\\
 * \mbox{to minimize: } & \mathbf f(\mathbf x, s) \in \mathbb R^{n_{obj}}\\
 * \mbox{subject to:} & \mathbf {c}_e(\mathbf x, s) = 0 \\
 *                    & \mathbf {c}_i(\mathbf x, s) \le 0
 * \end{array}
 * \f]
 *
 * where \f$\mathbf x \in \mathbb R^{n_x}\f$ is called *decision vector* or
 * *chromosome*, \f$\mathbf{lb}, \mathbf{ub} \in \mathbb R^{n_x}\f$ are the *box-bounds*,
 * \f$ \mathbf f: \mathbb R^{n_x} \rightarrow \mathbb R^{n_{obj}}\f$ define the *objectives*,
 * \f$ \mathbf c_e:  \mathbb R^{n_x} \rightarrow \mathbb R^{n_{ec}}\f$ are non linear *equality constraints*,
 * and \f$ \mathbf c_i:  \mathbb R^{n_x} \rightarrow \mathbb R^{n_{ic}}\f$ are non linear *inequality constraints*.
 * Note that the objectives and constraints may also depend from an added value \f$s\f$ seeding the
 * values of any number of stochastic variables. This allows also for stochastic programming
 * tasks to be represented by this class.
 *
 * To create an instance of the above problem the user is asked to construct a pagmo::problem from
 * a separate object of type \p T where, at least, the implementation of
 * the following methods is provided:
 *
 * @code
 * vector_double fitness(const decision_vector &) const;
 * vector_double::size_type get_nobj() const;
 * std::pair<vector_double, vector_double> get_bounds() const;
 * @endcode
 *
 * - The return value of \p T::fitness() is expected to have a dimension of \f$n_{f} = n_{obj} + n_{ec} + n_{ic}\f$
 * and to contain the concatenated values of \f$\mathbf f, \mathbf c_e\f$ and \f$\mathbf c_i\f$, (in this order).
 * - The return value of \p T::get_nobj() is expected to be \f$n_{obj}\f$
 * - The return value of \p T::get_bounds() is expected to contain \f$(\mathbf{lb}, \mathbf{ub})\f$.
 *
 * The three mandatory methods above allow to define a deterministic, derivative-free, unconstrained problem.
 * In order to consider more complex cases, the user may implement one or more
 * of the following methods in \p T :
 *   @code
 *   vector_double::size_type get_nec() const;
 *   vector_double::size_type get_nic() const;
 *   vector_double gradient(const vector_double &x) const;
 *   sparsity_pattern gradient_sparsity() const;
 *   std::vector<vector_double> hessians(const vector_double &x) const;
 *   std::vector<sparsity_pattern> hessians_sparsity() const;
 *   void set_seed(unsigned int s);
 *   std::string get_name() const;
 *   std::string get_extra_info() const;
 *   @endcode
 *
 * - \p T::get_nec() returns \f$n_{ec}\f$. When not implemented \f$n_{ec} = 0\f$ is assumed, and the pagmo::problem::get_nec() method will return 0.
 * - \p T::get_nic() returns \f$n_{ic}\f$. When not implemented \f$n_{ic} = 0\f$ is assumed, and the pagmo::problem::get_nic() method will return 0.
 * - \p T::gradient() returns a sparse representation of the gradients. The \f$ k\f$-th term
 * is expected to contain \f$ \frac{\partial f_i}{\partial x_j}\f$, where the pair \f$(i,j)\f$
 * is the \f$k\f$-th element of the sparsity pattern (collection of index pairs) as returned by problem::gradient_sparsity().
 * When not implemented, a call to problem::gradient() throws an \p std::logic_error.
 * - \p T::gradient_sparsity() returns the gradient sparsity pattern, i.e a collection of the non-zero index pairs \f$(i,j)\f$. When
 * not implemented a dense pattern is assumed and a call to problem::gradient_sparsity().
 * returns \f$((0,0),(0,1), ... (0,n_x-1), ...(n_f-1,n_x-1))\f$
 * - \p T::hessians() returns a vector of sparse representations for the hessians. For
 * the \f$l\f$-th value returned by \p T::fitness(), the hessian is defined as \f$ \frac{\partial f^2_l}{\partial x_i\partial x_j}\f$
 * and its sparse representation is in the \f$l\f$-th value returned by \p T::hessians(). Since
 * the hessians are symmetric, their sparse representation only contain lower triangular elements. The indexes
 * \f$(i,j)\f$ are stored in the \f$l\f$-th sparsity pattern (collection of index pairs) returned by problem::hessians_sparsity().
 * When not implemented, a call to problem::hessians() throws an \p std::logic_error.
 * - \p T::hessians_sparsity() returns an \p std::vector of sparsity patterns, each one being
 * a collection of the non-zero index pairs \f$(i,j)\f$ of the corresponding Hessian. Since the Hessian matrix
 * is symmetric, only lower triangular elements are allowed. When
 * not implemented a dense pattern is assumed and a call to problem::hessians_sparsity()
 * returns \f$n_f\f$ sparsity patterns each one being \f$((0,0),(1,0), (1,1), (2,0) ... (n_x-1,n_x-1))\f$.
 * - \p T::set_seed() changes the value of the seed \f$s\f$ that can be used in the fitness function to
 * consider stochastic objectives and constraints. When not implemented a call to problem::set_seed() throws an \p std::logic_error.
 * - \p T::get_name() returns a string containing the problem name to be used in output streams.
 * - \p T::get_extra_info() returns a string containing extra human readable information to be used in output streams.
 *
 * @note Three counters are defined in the class to keep track of evaluations of the fitness, the gradients and the hessians.
 * At each copy construction and copy assignment these counters are reset to zero.
 *
 * @note The only allowed operations on an object belonging to this class, after it has been moved, are assignment and destruction.
 */

class problem
{
        // Enable the generic ctor only if T is not a problem (after removing
        // const/reference qualifiers).
        template <typename T>
        using generic_ctor_enabler = std::enable_if_t<!std::is_same<problem,std::decay_t<T>>::value,int>;
        // Two helper functions to compute sparsity patterns in the dense case.
        static std::vector<sparsity_pattern> dense_hessians(vector_double::size_type f_dim, vector_double::size_type dim)
        {
            std::vector<sparsity_pattern> retval(f_dim);
            for (auto &Hs: retval) {
                for (decltype(dim) j = 0u; j<dim; ++j) {
                    for (decltype(dim) i = 0u; i<=j; ++i) {
                       Hs.push_back({j,i});
                    }
                }
            }
            return retval;
        }
        static sparsity_pattern dense_gradient(vector_double::size_type f_dim, vector_double::size_type dim)
        {
            sparsity_pattern retval;
            for (decltype(f_dim) j = 0u; j<f_dim; ++j) {
                for (decltype(dim) i = 0u; i<dim; ++i) {
                   retval.push_back({j,i});
                }
            }
            return retval;
        }
    public:
        /// Constructor from a user problem of type \p T
        /**
         * Construct a pagmo::problem with fitness dimension \f$n_f\f$ and decision vector
         * dimension \f$n_x\f$ from an object of type \p T. In
         * order for the construction to be successfull \p T needs
         * to satisfy the following requests:
         *
         * - \p T must implement the following mandatory methods:
         *   @code
         *   vector_double fitness(const decision_vector &) const;
         *   vector_double::size_type get_nobj() const;
         *   std::pair<vector_double, vector_double> get_bounds() const;
         *   @endcode
         *   otherwise it will result in a compile-time failure
         * - \p T must be not of type pagmo::problem, otherwise this templated constructor is not enabled
         * - \p T must be default-constructible, copy-constructible, move-constructible and destructible,
         *   otherwise it will result in a compile-time failure
         *
         * @note The fitness dimension \f$n_f = n_{obj} + n_{ec} + n_{ic}\f$ is defined by the return value of problem::get_nf(),
         * while the decision vector dimension \f$n_x\f$ is defined
         * by the size of the bounds as returned by \p T::get_bounds()
         *
         * @param[in] x The user implemented problem
         *
         * @throws std::invalid_argument If the upper and lower bounds returned by the mandatory method \p T::get_bounds() have different length.
         * @throws std::invalid_argument If the upper and lower bounds returned by the mandatory method \p T::get_bounds() are not such that \f$lb_i \le ub_i, \forall i\f$
         * @throws std::invalid_argument If \p T has a \p T::gradient_sparsity() method and this returns an invalid index pair \f$ (i,j)\f$ having \f$i \ge n_f\f$ or \f$j \ge n_x\f$
         * @throws std::invalid_argument If \p T has a \p T::gradient_sparsity() method and this contains any repeated index pair.
         * @throws std::invalid_argument If \p T has a \p T::hessians_sparsity() method and this returns an invalid index pair \f$ (i,j)\f$ having \f$i \ge n_x\f$ or \f$j > i\f$
         * @throws std::invalid_argument If \p T has a \p T::hessians_sparsity() method and this contains any repeated index pair.
         */
        template <typename T, generic_ctor_enabler<T> = 0>
        explicit problem(T &&x):m_ptr(::new detail::prob_inner<std::decay_t<T>>(std::forward<T>(x))),m_fevals(0u),m_gevals(0u),m_hevals(0u)
        {
            // 1 - Bounds.
            auto bounds = ptr()->get_bounds();
            detail::check_problem_bounds(bounds);
            m_lb = std::move(bounds.first);
            m_ub = std::move(bounds.second);
            // 2 - Number of objectives.
            m_nobj = ptr()->get_nobj();
            if (m_nobj == 0u) {
                pagmo_throw(std::invalid_argument,"The number of objectives must be at least 1, but "
                + std::to_string(m_nobj) + " was provided instead");
            }
            // NOTE: here we check that we can always compute nobj + nec + nic safely.
            if (m_nobj > std::numeric_limits<decltype(m_nobj)>::max() / 3u) {
                pagmo_throw(std::invalid_argument,"The number of objectives is too large");
            }
            // 3 - Constraints.
            m_nec = ptr()->get_nec();
            if (m_nec > std::numeric_limits<decltype(m_nec)>::max() / 3u) {
                pagmo_throw(std::invalid_argument,"The number of equality constraints is too large");
            }
            m_nic = ptr()->get_nic();
            if (m_nic > std::numeric_limits<decltype(m_nic)>::max() / 3u) {
                pagmo_throw(std::invalid_argument,"The number of inequality constraints is too large");
            }
            // 4 - Presence of gradient and its sparsity.
            m_has_gradient = ptr()->has_gradient();
            m_has_gradient_sparsity = ptr()->has_gradient_sparsity();
            // 5 - Presence of Hessians and their sparsity.
            m_has_hessians = ptr()->has_hessians();
            m_has_hessians_sparsity = ptr()->has_hessians_sparsity();
            // 5bis - Is this a stochastic problem?
            m_has_set_seed = ptr()->has_set_seed();
            // 6 - Name and extra info.
            m_name = ptr()->get_name();
            m_extra_info = ptr()->get_extra_info();
            // 7 - Check the sparsities, and cache their sizes.
            if (m_has_gradient_sparsity) {
                // If the problem provides gradient sparsity, get it, check it
                // and store its size.
                const auto gs = ptr()->gradient_sparsity();
                check_gradient_sparsity(gs);
                m_gs_dim = gs.size();
            } else {
                // If the problem does not provide gradient sparsity, we assume dense
                // sparsity. We can compute easily the expected size of the sparsity
                // in this case.
                const auto nx = get_nx();
                const auto nf = get_nf();
                if (nx > std::numeric_limits<vector_double::size_type>::max() / nf) {
                    pagmo_throw(std::invalid_argument,"The size of the (dense) gradient "
                        "sparsity is too large");
                }
                m_gs_dim = nx * nf;
            }
            // Same as above for the hessians.
            if (m_has_hessians_sparsity) {
                const auto hs = ptr()->hessians_sparsity();
                check_hessians_sparsity(hs);
                for (const auto &one_hs: hs) {
                    m_hs_dim.push_back(one_hs.size());
                }
            } else {
                const auto nx = get_nx();
                const auto nf = get_nf();
                if (nx == std::numeric_limits<vector_double::size_type>::max() ||
                    nx / 2u > std::numeric_limits<vector_double::size_type>::max() / (nx + 1u))
                {
                    pagmo_throw(std::invalid_argument,"The size of the (dense) hessians "
                        "sparsity is too large");
                }
                for (vector_double::size_type i = 0u; i < nf; ++i) {
                    m_hs_dim.push_back(nx * (nx - 1u) / 2u + nx); // lower triangular
                }
            }
        }

        /// Copy constructor
        problem(const problem &other):
            m_ptr(other.ptr()->clone()),
            m_fevals(0u),m_gevals(0u),m_hevals(0u),
            m_lb(other.m_lb),m_ub(other.m_ub),m_nobj(other.m_nobj),
            m_nec(other.m_nec),m_nic(other.m_nic),
            m_has_gradient(other.m_has_gradient),m_has_gradient_sparsity(other.m_has_gradient_sparsity),
            m_has_hessians(other.m_has_hessians),m_has_hessians_sparsity(other.m_has_hessians_sparsity),
            m_has_set_seed(other.m_has_set_seed), m_name(other.m_name),m_extra_info(other.m_extra_info),
            m_gs_dim(other.m_gs_dim),m_hs_dim(other.m_hs_dim)
        {}

        /// Move constructor
        problem(problem &&other) noexcept :
            m_ptr(std::move(other.m_ptr)),
            m_fevals(other.m_fevals.load()),
            m_gevals(other.m_gevals.load()),
            m_hevals(other.m_hevals.load()),
            m_lb(std::move(other.m_lb)),m_ub(std::move(other.m_ub)),m_nobj(other.m_nobj),
            m_nec(other.m_nec),m_nic(other.m_nic),
            m_has_gradient(other.m_has_gradient),m_has_gradient_sparsity(other.m_has_gradient_sparsity),
            m_has_hessians(other.m_has_hessians),m_has_hessians_sparsity(other.m_has_hessians_sparsity),
            m_has_set_seed(other.m_has_set_seed), m_name(std::move(other.m_name)),m_extra_info(std::move(other.m_extra_info)),
            m_gs_dim(other.m_gs_dim),m_hs_dim(std::move(other.m_hs_dim))
        {}

        /// Move assignment operator
        problem &operator=(problem &&other) noexcept
        {
            if (this != &other) {
                m_ptr = std::move(other.m_ptr);
                m_fevals.store(other.m_fevals.load());
                m_gevals.store(other.m_gevals.load());
                m_hevals.store(other.m_hevals.load());
                m_lb = std::move(other.m_lb);
                m_ub = std::move(other.m_ub);
                m_nobj = other.m_nobj;
                m_nec = other.m_nec;
                m_nic = other.m_nic;
                m_has_gradient = other.m_has_gradient;
                m_has_gradient_sparsity = other.m_has_gradient_sparsity;
                m_has_hessians = other.m_has_hessians;
                m_has_hessians_sparsity = other.m_has_hessians_sparsity;
                m_has_set_seed = other.m_has_set_seed,
                m_name = std::move(other.m_name);
                m_extra_info = std::move(other.m_extra_info);
                m_gs_dim = other.m_gs_dim;
                m_hs_dim = std::move(other.m_hs_dim);
            }
            return *this;
        }

        /// Copy assignment operator
        problem &operator=(const problem &other)
        {
            // Copy ctor + move assignment.
            return *this = problem(other);
        }

        /// Extracts the user-defined problem
        /**
         * Extracts the original problem that was provided by the user, thus
         * granting access to additional resources there implemented.
         *
         * @tparam T The type of the orignal user-defined problem
         *
         * @return a const pointer to the user-defined problem, or \p nullptr
         * if \p T does not correspond exactly to the original problem type used
         * in the constructor.
         */
        template <typename T>
        const T *extract() const
        {
            auto p = dynamic_cast<const detail::prob_inner<T> *>(ptr());
            if (p == nullptr) {
                return nullptr;
            }
            return &(p->m_value);
        }

        /// Checks the user defined problem type at run-time
        /**
         * @tparam T The type to be checked
         *
         * @return \p true if the user defined problem is \p T, \p false othewise.
         */
        template <typename T>
        bool is() const
        {
            return extract<T>() != nullptr;
        }

        /// Computes the fitness
        /**
         * The fitness, implemented in the user-defined problem,
         * is expected to be a pagmo::vector_double of dimension \f$n_f\f$ containing
         * the problem fitness: the concatenation of \f$n_{obj}\f$ objectives
         * to minimize, \f$n_{ec}\f$ equality constraints and \f$n_{ic}\f$
         * inequality constraints.
         *
         * @param[in] dv The decision vector
         *
         * @return The user implemented fitness.
         *
         * @throws std::invalid_argument if the length of the decision vector is not \f$n_x\f$
         * @throws std::invalid_argument if the length of the fitness returned (as defined in the user defined problem)
         * is not \f$n_f\f$
         */
        vector_double fitness(const vector_double &dv) const
        {
            // 1 - checks the decision vector
            check_decision_vector(dv);
            // 2 - computes the fitness
            vector_double retval(ptr()->fitness(dv));
            // 3 - checks the fitness vector
            check_fitness_vector(retval);
            // 4 - increments fitness evaluation counter
            ++m_fevals;
            return retval;
        }

        /// Computes the gradient
        /**
         * The gradient, optionally implemented in the user-defined problem,
         * is expected to be a pagmo::vector_double containing the problem
         * fitness gradients \f$ g_{ij} = \frac{\partial f_i}{\partial x_j}\f$
         * in the order specified by the gradient sparsity pattern returned by
         * problem::gradient_sparsity()
         * (a vector of index pairs \f$(i,j)\f$).
         *
         * @param[in] dv The decision vector
         *
         * @return The gradient as implemented by the user.
         *
         * @throws std::invalid_argument if the length of the decision vector \p dv is not \f$n_x\f$
         * @throws std::invalid_argument if the length of the gradient returned (as defined in the user defined problem)
         * does not match the gradient sparsity pattern dimension as returned by
         * problem::get_gs_dim()
         * @throws std::logic_error if the user defined problem does not implement
         * the gradient method
         */
        vector_double gradient(const vector_double &dv) const
        {
            // 1 - checks the decision vector
            check_decision_vector(dv);
            // 2 - compute the gradients
            vector_double retval(ptr()->gradient(dv));
            // 3 - checks the gradient vector
            check_gradient_vector(retval);
            // 4 - increments gradient evaluation counter
            ++m_gevals;
            return retval;
        }

        /// Checks if the user-defined problem has a gradient
        /**
         * If the user defined problem implements a gradient, this
         * will return true, false otherwise. The value returned can
         * also be directly hard-coded implementing the
         * method
         *
         * @code
         * bool has_gradient() const
         * @endcode
         *
         * in the user-defined problem
         *
         * @return a boolean flag
         */
        bool has_gradient() const
        {
            return m_has_gradient;
        }

        /// Computes the gradient sparsity pattern
        /**
         * The gradient sparsity pattern is a collection of the indexes
         * \f$(i,j)\f$ of the non-zero elements of
         * \f$ g_{ij} = \frac{\partial f_i}{\partial x_j}\f$. By default
         * PaGMO assumes a dense pattern (all index pairs in the order
         * \f$(0,0) .. (0,n_x-1), ...(1,0) .. (1,n_x-1) .. (n_f-1,n_x-1)\f$
         * but this default is overidden if the method gradient_sparsity is
         * implemented in the user defined problem.
         *
         * @return The gradient sparsity pattern.
         */
        sparsity_pattern gradient_sparsity() const
        {
            if (has_gradient_sparsity()) {
                auto retval = ptr()->gradient_sparsity();
                check_gradient_sparsity(retval);
                return retval;
            }
            return dense_gradient(get_nf(),get_nx());
        }

        /// Checks if the user-defined problem has a gradient_sparsity
        /**
         * If the user defined problem implements a gradient_sparsity, this
         * will return true, false otherwise. The value returned can
         * also be directly hard-coded implementing the
         * method
         *
         * @code
         * bool has_gradient_sparsity() const
         * @endcode
         *
         * in the user-defined problem
         *
         * @return a boolean flag
         */
        bool has_gradient_sparsity() const
        {
            return m_has_gradient_sparsity;
        }

        /// Computes the hessians
        /**
         * The hessians, optionally implemented in the user-defined problem,
         * are expected to be an <tt>std::vector</tt> of pagmo::vector_double.
         * The element \f$ l\f$ contains the problem hessian:
         * \f$ h^l_{ij} = \frac{\partial f^2_l}{\partial x_i\partial x_j}\f$
         * in the order specified by the \f$ l\f$-th element of the
         * hessians sparsity pattern (a vector of index pairs \f$(i,j)\f$)
         * as returned by problem::hessians_sparsity()
         *
         * @param[in] dv The decision vector
         *
         * @return The hessians as implemented by the user.
         *
         * @throws std::invalid_argument if the length of the decision vector \p dv is not \f$n_x\f$
         * @throws std::invalid_argument if the length of each hessian returned
         * (as defined in the user defined problem) does not match the corresponding
         * hessians sparsity pattern dimensions as returned by problem::get_hs_dim()
         * @throws std::logic_error if the user defined problem does not implement
         * the hessians method
         */
        std::vector<vector_double> hessians(const vector_double &dv) const
        {
            // 1 - checks the decision vector
            check_decision_vector(dv);
            // 2 - computes the hessians
            std::vector<vector_double> retval(ptr()->hessians(dv));
            // 3 - checks the hessians
            check_hessians_vector(retval);
            // 4 - increments hessians evaluation counter
            ++m_hevals;
            return retval;
        }

        /// Check if the user-defined problem implements the hessians
        /**
         * If the user defined problem implements hessians, this
         * will return true, false otherwise. The value returned can
         * also be directly hard-coded implementing the
         * method
         *
         * @code
         * bool has_hessians() const
         * @endcode
         *
         * in the user-defined problem
         *
         * @return a boolean flag
         */
        bool has_hessians() const
        {
            return m_has_hessians;
        }

        /// Computes the hessians sparsity pattern
        /**
         * Each component \f$ l\f$ of the hessians sparsity pattern is a
         * collection of the indexes \f$(i,j)\f$ of the non-zero elements of
         * \f$h^l_{ij} = \frac{\partial f^l}{\partial x_i\partial x_j}\f$. By default
         * PaGMO assumes a dense pattern storing a lower triangular representation
         * (all index pairs in the order
         * \f$(0,0), (1,0), (1,1), (2,0) ... (n_x-1,n_x-1)\f$
         * but this default is overidden if the method hessians_sparsity is
         * implemented in the user defined problem.
         *
         * @return The hessians sparsity pattern.
         */
        std::vector<sparsity_pattern> hessians_sparsity() const
        {
            if (has_hessians_sparsity()) {
                auto retval = ptr()->hessians_sparsity();
                check_hessians_sparsity(retval);
                return retval;
            }
            return dense_hessians(get_nf(),get_nx());
        }

        /// Check if the user-defined dproblem implements the hessians_sparsity
        /**
         * If the user defined problem implements a hessians sparsity, this
         * will return true, false otherwise. The value returned can
         * also be directly hard-coded implementing the
         * method
         *
         * @code
         * bool has_hessians_sparsity() const
         * @endcode
         *
         * in the user-defined problem
         *
         * @return a boolean flag
         *
         */
        bool has_hessians_sparsity() const
        {
            return m_has_hessians_sparsity;
        }

        /// Number of objectives
        /**
         * @return Returns \f$ n_{obj}\f$, the number of objectives as returned by the
         * corresponding user-implemented method
         */
        vector_double::size_type get_nobj() const
        {
            return m_nobj;
        }

        /// Problem dimension
        /**
         * @return Returns \f$ n_{x}\f$, the dimension of the decision vector as implied
         * by the length of the bounds returned by the user-implemented get_bounds method
         */
        vector_double::size_type get_nx() const
        {
            return m_lb.size();
        }

        /// Fitness dimension
        /**
         * @return Returns \f$ n_{f}\f$, the dimension of the fitness as the
         * sum of \f$n_{obj}\f$, \f$n_{ec}\f$, \f$n_{ic}\f$
         */
        vector_double::size_type get_nf() const
        {
            return m_nobj + m_nic + m_nec;
        }

        /// Box-bounds
        /**
         * @return Returns \f$ (\mathbf{lb}, \mathbf{ub}) \f$, the box-bounds as returned by
         * the corresponding user-implemented method
         */
        std::pair<vector_double, vector_double> get_bounds() const
        {
            return std::make_pair(m_lb,m_ub);
        }

        /// Number of equality constraints
        /**
         * @return Returns \f$ n_{ec} \f$, the number of inequality constraints
         * as returned by the the corresponding user-implemented method if present,
         * zero otherwise
         */
        vector_double::size_type get_nec() const
        {
            return m_nec;
        }

        /// Number of inequality constraints
        /**
         * @return Returns \f$ n_{ic} \f$, the number of inequality constraints
         * as returned by the the corresponding user-implemented method if present,
         * zero otherwise
         */
        vector_double::size_type get_nic() const
        {
            return m_nic;
        }

        /// Number of constraints
        /**
         * @return Returns \f$ n_{ic} + n_{ec} \f$, the number of constraints
         */
        vector_double::size_type get_nc() const
        {
            return m_nec + m_nic;
        }

        /// Number of fitness evaluations
        unsigned long long get_fevals() const
        {
            return m_fevals.load();
        }

        /// Number of gradient evaluations
        unsigned long long get_gevals() const
        {
            return m_gevals.load();
        }

        /// Number of hessians evaluations
        unsigned long long get_hevals() const
        {
            return m_hevals.load();
        }

        /// Dimension of the gradient sparisy
        vector_double::size_type get_gs_dim() const
        {
            return m_gs_dim;
        }

        /// Dimension of the hessians sparisy
        std::vector<vector_double::size_type> get_hs_dim() const
        {
            return m_hs_dim;
        }

        /// Sets the seed for the stochastic variables
        /**
         * Sets the seed to be used in the fitness function to instantiate
         * all stochastic variables.
         *
         * @param[in] seed seed
         */
        void set_seed(unsigned int seed)
        {
            ptr()->set_seed(seed);
        }

        /// Check if the user-defined problem implements a set_seed method
        /**
         * If the user defined problem implements a set_seed method, this
         * will return true, false otherwise. The value returned can
         * also be forced by the user by implementing the additional
         * method
         *
         * @code
         * bool has_set_seed() const
         * @endcode
         *
         * in the user-defined problem
         *
         * @return a boolean flag
         *
         */
        bool has_set_seed() const
        {
            return m_has_set_seed;
        }

        /// Check if the user-defined problem implements a set_seed method
        /**
         * This method is an alias for problem::has_set_seed().
         * If the user defined problem implements a set_seed method, this
         * will return true, false otherwise. The value returned can
         * also be forced by the user by implementing the additional
         * method
         *
         * @code
         * bool has_set_seed() const
         * @endcode
         *
         * in the user-defined problem
         *
         * @return a boolean flag
         *
         */
        bool is_stochastic() const
        {
            return has_set_seed();
        }

        /// Problem's name.
        /**
         * @return The problem's name as returned by the corresponding
         * user-implemented method if present, the C++ mingled class name otherwise.
         */
        std::string get_name() const
        {
            return m_name;
        }

        /// Extra info
        /**
         * @return The problem's extra info as returned by the corresponding
         * user-implemented method if present, an empty string otehrwise.
         */
        std::string get_extra_info() const
        {
            return m_extra_info;
        }

        /// Streaming operator
        /**
         * @return An std::ostream containing a human-readable
         * representation of the problem, includeing the result from
         * the user-defined method extra_info if implemented.
         */
        friend std::ostream &operator<<(std::ostream &os, const problem &p)
        {
            os << "Problem name: " << p.get_name();
            if (p.is_stochastic()) {
                stream(os, " [stochastic]");
            }
            os << "\n\tGlobal dimension:\t\t\t" << p.get_nx() << '\n';
            os << "\tFitness dimension:\t\t\t" << p.get_nf() << '\n';
            os << "\tNumber of objectives:\t\t\t" << p.get_nobj() << '\n';
            os << "\tEquality constraints dimension:\t\t" << p.get_nec() << '\n';
            os << "\tInequality constraints dimension:\t" << p.get_nic() << '\n';
            os << "\tLower bounds: ";
            stream(os, p.get_bounds().first, '\n');
            os << "\tUpper bounds: ";
            stream(os, p.get_bounds().second, '\n');
            stream(os, "\n\tHas gradient: ", p.has_gradient(), '\n');
            stream(os, "\tUser implemented gradient sparsity: ", p.has_gradient_sparsity(), '\n');
            if (p.has_gradient()) {
                stream(os, "\tExpected gradients: ", p.m_gs_dim, '\n');
            }
            stream(os, "\tHas hessians: ", p.has_hessians(), '\n');
            stream(os, "\tUser implemented hessians sparsity: ", p.has_hessians_sparsity(), '\n');
            if (p.has_hessians()) {
                stream(os, "\tExpected hessian components: ", p.m_hs_dim, '\n');
            }
            stream(os, "\n\tFunction evaluations: ", p.get_fevals(), '\n');
            if (p.has_gradient()) {
                stream(os, "\tGradient evaluations: ", p.get_gevals(), '\n');
            }
            if (p.has_hessians()) {
                stream(os, "\tHessians evaluations: ", p.get_hevals(), '\n');
            }

            const auto extra_str = p.get_extra_info();
            if (!extra_str.empty()) {
                stream(os, "\nExtra info:\n", extra_str);
            }
            return os;
        }

        /// Serialization: save
        template <typename Archive>
        void save(Archive &ar) const
        {
            ar(m_ptr,m_fevals.load(), m_gevals.load(), m_hevals.load(),
                m_lb,m_ub,m_nobj,m_nec,m_nic,m_has_gradient,m_has_gradient_sparsity,
                m_has_hessians,m_has_hessians_sparsity,m_has_set_seed,m_name,m_extra_info,
                m_gs_dim,m_hs_dim);
        }

        /// Serialization: load
        template <typename Archive>
        void load(Archive &ar)
        {
            ar(m_ptr);
            unsigned long long tmp;
            ar(tmp);
            m_fevals.store(tmp);
            ar(tmp);
            m_gevals.store(tmp);
            ar(tmp);
            m_hevals.store(tmp);
            ar(m_lb,m_ub,m_nobj,m_nec,m_nic,m_has_gradient,m_has_gradient_sparsity,
                m_has_hessians,m_has_hessians_sparsity,m_has_set_seed,m_name,m_extra_info,
                m_gs_dim,m_hs_dim);
        }

    private:
        // Just two small helpers to make sure that whenever we require
        // access to the pointer it actually points to something.
        detail::prob_inner_base const *ptr() const
        {
            assert(m_ptr.get() != nullptr);
            return m_ptr.get();
        }
        detail::prob_inner_base *ptr()
        {
            assert(m_ptr.get() != nullptr);
            return m_ptr.get();
        }

        // A small helper to check if a vector containes unique elements.
        template <typename U>
        static bool all_unique(std::vector<U> x)
        {
            std::sort(x.begin(),x.end());
            auto it = std::unique(x.begin(),x.end());
            return it == x.end();
        }

        void check_gradient_sparsity(const sparsity_pattern &gs) const
        {
            const auto nx = get_nx();
            const auto nf = get_nf();
            // 1 - We check that the gradient sparsity pattern has
            // valid indexes.
            for (const auto &pair: gs) {
                if ((pair.first >= nf) || (pair.second >= nx)) {
                    pagmo_throw(std::invalid_argument,"Invalid pair detected in the gradient sparsity pattern: (" + std::to_string(pair.first) + ", " + std::to_string(pair.second) + ")\nFitness dimension is: " + std::to_string(nf) + "\nDecision vector dimension is: " + std::to_string(nx));
                }
            }
            // 2 - We check all pairs are unique
            if (!all_unique(gs)) {
                pagmo_throw(std::invalid_argument,"Multiple entries of the same index pair was detected in the gradient sparsity pattern");
            }
        }
        void check_hessians_sparsity(const std::vector<sparsity_pattern> &hs) const
        {
            // 1 - We check that a hessian sparsity is provided for each component
            // of the fitness
            const auto nf = get_nf();
            if (hs.size()!=nf) {
                pagmo_throw(std::invalid_argument,"Invalid dimension of the hessians_sparsity: " + std::to_string(hs.size()) + ", expected: " + std::to_string(nf));
            }
            // 2 - We check that all hessian sparsity patterns have
            // valid indexes.
            for (const auto &one_hs: hs) {
                check_hessian_sparsity(one_hs);
            }
        }
        void check_hessian_sparsity(const sparsity_pattern &hs) const
        {
            const auto nx = get_nx();
            // 1 - We check that the hessian sparsity pattern has
            // valid indexes. Assuming a lower triangular representation of
            // a symmetric matrix. Example, for a 4x4 dense symmetric
            // [(0,0), (1,0), (1,1), (2,0), (2,1), (2,2), (3,0), (3,1), (3,2), (3,3)]
            for (const auto &pair: hs) {
                if ((pair.first >= nx) || (pair.second > pair.first)) {
                    pagmo_throw(std::invalid_argument,"Invalid pair detected in the hessians sparsity pattern: (" + std::to_string(pair.first) + ", " + std::to_string(pair.second) + ")\nDecision vector dimension is: " + std::to_string(nx) + "\nNOTE: hessian is a symmetric matrix and PaGMO represents it as lower triangular: i.e (i,j) is not valid if j>i");
                }
            }
            // 2 -  We check all pairs are unique
            if (!all_unique(hs)) {
                pagmo_throw(std::invalid_argument, "Multiple entries of the same index pair were detected in the hessian sparsity pattern");
            }
        }
        void check_decision_vector(const vector_double &dv) const
        {
            // 1 - check decision vector for length consistency
            if (dv.size()!=get_nx()) {
                pagmo_throw(std::invalid_argument,"Length of decision vector is " + std::to_string(dv.size()) + ", should be " + std::to_string(get_nx()));
            }
            // 2 - Here is where one could check if the decision vector
            // is in the bounds. At the moment not implemented
        }

        void check_fitness_vector(const vector_double &f) const
        {
            auto nf = get_nf();
            // Checks dimension of returned fitness
            if (f.size()!=nf) {
                pagmo_throw(std::invalid_argument,"Fitness length is: " + std::to_string(f.size()) + ", should be " + std::to_string(nf));
            }
        }

        void check_gradient_vector(const vector_double &gr) const
        {
            // Checks that the gradient vector returned has the same dimensions of the sparsity_pattern
            if (gr.size()!=m_gs_dim) {
                pagmo_throw(std::invalid_argument,"Gradients returned: " + std::to_string(gr.size()) + ", should be " + std::to_string(m_gs_dim));
            }
        }

        void check_hessians_vector(const std::vector<vector_double> &hs) const
        {
            // Checks that the hessians returned have the same dimensions of the
            // corresponding sparsity patterns
            for (decltype(hs.size()) i=0u; i<hs.size(); ++i) {
                if (hs[i].size()!=m_hs_dim[i]) {
                    pagmo_throw(std::invalid_argument,"On the hessian no. " + std::to_string(i) +  ": Components returned: " + std::to_string(hs[i].size()) + ", should be " + std::to_string(m_hs_dim[i]));
                }
            }
        }

    private:
        // Pointer to the inner base problem
        std::unique_ptr<detail::prob_inner_base> m_ptr;
        // Atomic counter for calls to the fitness
        mutable std::atomic<unsigned long long> m_fevals;
        // Atomic counter for calls to the gradient
        mutable std::atomic<unsigned long long> m_gevals;
        // Atomic counter for calls to the hessians
        mutable std::atomic<unsigned long long> m_hevals;
        // Various problem properties determined at construction time
        // from the concrete problem. These will be constant for the lifetime
        // of problem, but we cannot mark them as such because of serialization.
        vector_double m_lb;
        vector_double m_ub;
        vector_double::size_type m_nobj;
        vector_double::size_type m_nec;
        vector_double::size_type m_nic;
        bool m_has_gradient;
        bool m_has_gradient_sparsity;
        bool m_has_hessians;
        bool m_has_hessians_sparsity;
        bool m_has_set_seed;
        std::string m_name;
        std::string m_extra_info;
        // These are the dimensions of the sparsity objects, cached
        // here upon construction in order to provide fast checking
        // on the returned gradient and hessians.
        vector_double::size_type m_gs_dim;
        std::vector<vector_double::size_type> m_hs_dim;
};

} // namespaces

#endif
