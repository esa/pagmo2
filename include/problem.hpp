#ifndef PAGMO_PROBLEM_HPP
#define PAGMO_PROBLEM_HPP

#include <algorithm>
#include <atomic>
#include <iostream>
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

#define PAGMO_REGISTER_PROBLEM(prob) CEREAL_REGISTER_TYPE_WITH_NAME(pagmo::detail::prob_inner<prob>,#prob)

namespace pagmo
{

namespace detail
{

inline std::vector<sparsity_pattern> dense_hessians(vector_double::size_type f_dim, vector_double::size_type dim)
{
    std::vector<sparsity_pattern> retval(f_dim);
    for (auto &Hs : retval) {
        for (decltype(dim) j = 0u; j<dim; ++j) {
            for (decltype(dim) i = 0u; i<=j; ++i) {
               Hs.push_back({j,i});
            }
        }
    }
    return retval;
}

inline sparsity_pattern dense_gradient(vector_double::size_type f_dim, vector_double::size_type dim)
{
    sparsity_pattern retval;
    for (decltype(f_dim) j = 0u; j<f_dim; ++j) {
        for (decltype(dim) i = 0u; i<dim; ++i) {
           retval.push_back({j, i});
        }
    }
    return retval;
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
        "A problem must provide a fitness function [vector_double fitness(const vector_double &x) const] and a method to query the number of objectives [vector_double::size_type get_nobj() const].");
    static_assert(has_bounds<T>::value,
        "A problem must provide getters for its bounds [std::pair<vector_double, vector_double> get_bounds() const].");
    prob_inner() = default;
    prob_inner(const prob_inner &) = delete;
    prob_inner(prob_inner &&) = delete;
    explicit prob_inner(T &&x):m_value(std::move(x)) {}
    explicit prob_inner(const T &x):m_value(x) {}
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
    // Optional methods.
    template <typename U, typename std::enable_if<pagmo::has_gradient<U>::value,int>::type = 0>
    static vector_double gradient_impl(U &value, const vector_double &dv)
    {
        return value.gradient(dv);
    }
    template <typename U, typename std::enable_if<!pagmo::has_gradient<U>::value,int>::type = 0>
    static vector_double gradient_impl(U &, const vector_double &)
    {
        pagmo_throw(std::logic_error,"Gradients have been requested but not implemented.\n"
            "A function with prototype 'vector_double gradient(const vector_double &x)' const was expected.");
    }
    template <typename U, typename std::enable_if<pagmo::has_gradient<U>::value,int>::type = 0>
    static bool has_gradient_impl(U &)
    {
       return true;
    }
    template <typename U, typename std::enable_if<!pagmo::has_gradient<U>::value,int>::type = 0>
    static bool has_gradient_impl(U &)
    {
       return false;
    }
    template <typename U, typename std::enable_if<pagmo::has_gradient_sparsity<U>::value,int>::type = 0>
    static sparsity_pattern gradient_sparsity_impl(const U &value)
    {
        return value.gradient_sparsity();
    }
    template <typename U, typename std::enable_if<!pagmo::has_gradient_sparsity<U>::value,int>::type = 0>
    sparsity_pattern gradient_sparsity_impl(const U &) const
    {
        // By default a problem is dense
        auto dim = get_bounds().first.size();
        auto f_dim = get_nobj() + get_nec() + get_nic();
        return dense_gradient(f_dim, dim);
    }
    template <typename U, typename std::enable_if<pagmo::has_gradient_sparsity<U>::value,int>::type = 0>
    static bool has_gradient_sparsity_impl(U &)
    {
       return true;
    }
    template <typename U, typename std::enable_if<!pagmo::has_gradient_sparsity<U>::value,int>::type = 0>
    static bool has_gradient_sparsity_impl(U &)
    {
       return false;
    }
    template <typename U, typename std::enable_if<pagmo::has_hessians<U>::value,int>::type = 0>
    static std::vector<vector_double> hessians_impl(U &value, const vector_double &dv)
    {
        return value.hessians(dv);
    }
    template <typename U, typename std::enable_if<!pagmo::has_hessians<U>::value,int>::type = 0>
    static std::vector<vector_double> hessians_impl(U &, const vector_double &)
    {
        pagmo_throw(std::logic_error,"Hessians have been requested but not implemented.\n"
            "A function with prototype 'std::vector<vector_double> hessians(const vector_double &x)' const was expected.");
    }
    template <typename U, typename std::enable_if<pagmo::has_hessians<U>::value,int>::type = 0>
    static bool has_hessians_impl(U &)
    {
       return true;
    }
    template <typename U, typename std::enable_if<!pagmo::has_hessians<U>::value,int>::type = 0>
    static bool has_hessians_impl(U &)
    {
       return false;
    }
    template <typename U, typename std::enable_if<pagmo::has_hessians_sparsity<U>::value,int>::type = 0>
    static std::vector<sparsity_pattern> hessians_sparsity_impl(const U &value)
    {
        return value.hessians_sparsity();
    }
    template <typename U, typename std::enable_if<!pagmo::has_hessians_sparsity<U>::value,int>::type = 0>
    std::vector<sparsity_pattern> hessians_sparsity_impl(const U &) const
    {
        // By default a problem has dense hessians
        auto dim = get_bounds().first.size();
        auto f_dim = get_nobj() + get_nec() + get_nic();
        return dense_hessians(f_dim, dim);
    }
    template <typename U, typename std::enable_if<pagmo::has_hessians_sparsity<U>::value,int>::type = 0>
    static bool has_hessians_sparsity_impl(U &)
    {
       return true;
    }
    template <typename U, typename std::enable_if<!pagmo::has_hessians_sparsity<U>::value,int>::type = 0>
    static bool has_hessians_sparsity_impl(U &)
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
    virtual std::string get_name() const override final
    {
        return get_name_impl(m_value);
    }
    virtual std::string get_extra_info() const override final
    {
        return get_extra_info_impl(m_value);
    }
    // Serialization.
    template <typename Archive>
    void serialize(Archive &ar)
    {
        ar(cereal::base_class<prob_inner_base>(this),m_value);
    }
    T m_value;
};

}

/// Problem class.
/**
 * This class represents a generic *mathematical programming* or *evolutionary* problem in the form:
 * \f[
 * \begin{array}{rl}
 * \mbox{find:}      & \mathbf {lb} \le \mathbf x \le \mathbf{ub}\\
 * \mbox{to minimize: } & \mathbf f(\mathbf x) \in \mathbb R^{n_f}\\
 * \mbox{subject to:} & \mathbf {c}_e(\mathbf x) = 0 \\
 *                    & \mathbf {c}_i(\mathbf x) \le 0 
 * \end{array}
 * \f]
 *
 * where \f$\mathbf x \in \mathbb R^{n_x}\f$ is called *decision vector* or
 * *chromosome*, \f$\mathbf{lb}, \mathbf{ub} \in \mathbb R^{n_x}\f$ are the *box-bounds*,
 * \f$ \mathbf f: \mathbb R^{n_x} \rightarrow \mathbb R^{n_f}\f$ define the *objectives*,
 * \f$ \mathbf c_e:  \mathbb R^{n_x} \rightarrow \mathbb R^{n_{ce}}\f$ are non linear *equality constraints*,
 * and \f$ \mathbf c_i:  \mathbb R^{n_x} \rightarrow \mathbb R^{n_{ci}}\f$ are non linear *inequality constraints*.
 *
 * To specify the details of the above problem the user is asked to construct this
 * object with from a separate class \p T providing at least the implementation of
 * the following methods:
 *
 * @code 
 * fitness_vector fitness(const decision_vector &) const;
 * vector_double::size_type get_nobj() const;
 * std::pair<vector_double, vector_double> get_bounds() const;
 * @endcode
 * 
 * - The return value of \p T::fitness is expected to have a dimension of \f$n_{f} + n_{ce} + n_{ci}\f$
 * and to contain the concatenated values of \f$\mathbf f, \mathbf c_e\f$ (in this order).
 * and \f$\mathbf c_i\f$. 
 * - The return value of \p T::get_nobj is expected to be
 * \f$n_f\f$.
 * - The return value of \p T::get_bounds is expected to contain
 * \f$(\mathbf{lb}, \mathbf{ub})\f$. By default, both \f$n_{ce}\f$ and \f$n_{ci}\f$ 
 * are set to zero.
 *
 * The user can also implement the following methods in \p T:
 *   @code 
 *   vector_double::size_type get_nec() const;
 *   vector_double::size_type get_nic() const;
 *   vector_double gradient(const vector_double &x) const;
 *   sparsity_pattern gradient_sparsity() const;
 *   std::vector<vector_double> hessians(const vector_double &x) const;
 *   std::vector<sparsity_pattern> hessians_sparsity() const;
 *   std::string get_name() const; 
 *   std::string get_extra_info() const;
 *   @endcode
 *
 * - \p T::get_nec returns \f$n_{ec}\f$. When not implemented \f$n_{ec} = 0\f$ is assumed.
 * - \p T::get_nic returns \f$n_{ic}\f$. When not implemented \f$n_{ic} = 0\f$ is assumed.
 * - \p T::gradient returns a sparse representation of the gradients. The \f$ k\f$-th term 
 * is expected to contain \f$ \frac{\partial f_i}{\partial x_j}\f$, where the pair \f$(i,j)\f$
 * is the \f$k\f$-th element of the sparsity pattern (collection of index pairs) returned by \p T::gradient_sparsity.
 * When not implemented, gradient based techniques will have to either determine these numerically
 * or complain.
 * - \p T::hessians returns a vector of sparse representations for the hessians. For
 * the \f$l\f$-th value returned by \p T::fitness, the hessian is defined as \f$ \frac{\partial f^2_l}{\partial x_i\partial x_j}\f$
 * and its sparse representation is in the \f$l\f$-th value returned by T::hessians. Since
 * the hessians are symmetric, their sparse representation only contain lower triangular elements. The indexes 
 * \f$(i,j)\f$ are stored in the \f$l\f$-th sparsity pattern (collection of index pairs) returned by \p T::hessians_sparsity
 * When not implemented, hessians based techniques will have to either determine these numerically
 * or complain.
 *
 * Three counters are defined in the class to keep track of evaluations of the fitness, the gradients and the hessians. At
 * each construction and copy these counters are reset to zero.
 *
 * The only allowed operations on objects of this class after they have been moved, are assignment and destruction.
 *
 * @author Francesco Biscani (bluescarni@gmail.com)
 * @author Dario Izzo (darioizzo@gmail.com)
 */
class problem
{
        // Enable the generic ctor only if T is not a problem (after removing
        // const/reference qualifiers).
        template <typename T>
        using generic_ctor_enabler = std::enable_if_t<!std::is_same<problem,std::decay_t<T>>::value,int>;
    public:
        /// Constructor from a user defined object \p T
        /**
         * Construct a pagmo::problem with fitness dimension \f$n_f\f$ and decision vector
         * dimension \f$n_x\f$ from an object \p T. In
         * order for the construction to be successfull \p T needs
         * to satisfy the following requests:
         *
         * - \p T must implement the following mandatory methods:
         *   @code 
         *   fitness_vector fitness(const decision_vector &) const;
         *   vector_double::size_type get_nobj() const;
         *   std::pair<vector_double, vector_double> get_bounds() const;
         *   @endcode
         *   otherwise compilation will result in a static_assert failiure
         * - \p T must be not of type pagmo::problem, otherwise this templated constructor is not enabled
         * - \p T must be default-constructible, copy-constructible, move-constructible and destructible,
         *   otherwise compilation will result in a static_assert failiure
         *
         * The following methods, if implemented in \p T, will override 
         * default choices:
         *   @code 
         *   vector_double::size_type get_nec() const;
         *   vector_double::size_type get_nic() const;
         *   vector_double gradient(const vector_double &x) const;
         *   sparsity_pattern gradient_sparsity(const vector_double &x) const;
         *   std::vector<vector_double> hessians(const vector_double &x) const;
         *   std::vector<sparsity_pattern> hessians_sparsity() const;
         *   std::string get_name() const;
         *   std::string get_extra_info() const;
         *   @endcode
         *
         * @note The fitness dimension \f$n_f\f$ is defined by the return value of 
         * \p T::get_nobj, while the decision vector dimension \f$n_x\f$ is defined 
         * by the size of the bounds as returned by \p T::get_bounds()
         *
         * @param[in] x The user implemented problem
         * 
         * @throws std::invalid_argument If the upper and lower bounds returned by the mandatory method \p T::get_bounds() have different length.
         * @throws std::invalid_argument If the upper and lower bounds returned by the mandatory method \p T::get_bounds() are not such that \f$lb_i \le ub_i, \forall i\f$
         * @throws std::invalid_argument If \p T has a \p T::gradient_sparsity method and this returns an invalid index pair \f$ (i,j)\f$ having \f$i \ge n_f\f$ or \f$j \ge n_x\f$
         * @throws std::invalid_argument If \p T has a \p T::gradient_sparsity method and this contains any repeated index pair.
         * @throws std::invalid_argument If \p T has a \p T::hessians_sparsity method and this returns an invalid index pair \f$ (i,j)\f$ having \f$i \ge n_x\f$ or \f$j > i\f$
         * @throws std::invalid_argument If \p T has a \p T::hessians_sparsity method and this contains any repeated index pair.
         */
        template <typename T, generic_ctor_enabler<T> = 0>
        explicit problem(T &&x):m_ptr(::new detail::prob_inner<std::decay_t<T>>(std::forward<T>(x))),m_fevals(0u),m_gevals(0u),m_hevals(0u)
        {
            // check bounds consistency
            auto bounds = get_bounds();
            const auto &lb = bounds.first;
            const auto &ub = bounds.second;
            // 1 - check bounds have equal length
            if (lb.size()!=ub.size()) {
                pagmo_throw(std::invalid_argument,"Length of lower bounds vector is " + std::to_string(lb.size()) + ", length of upper bound is " + std::to_string(ub.size()));
            }
            // 2 - checks lower < upper for all values in lb, lb
            for (decltype(lb.size()) i=0u; i < lb.size(); ++i) {
                if (lb[i] > ub[i]) {
                    pagmo_throw(std::invalid_argument,"The lower bound at position " + std::to_string(i) + " is " + std::to_string(lb[i]) +
                        " while the upper bound has the smaller value " + std::to_string(ub[i]));
                }
            }
            // 4 - initialize problem dimension (must be before
            // check_gradient_sparsity and check_hessians_sparsity)
            m_n = lb.size();
            // 5 - checks that the sparsity contains reasonable numbers
            check_gradient_sparsity(); // here m_gs_dim is initialized
            // 6 - checks that the hessians contain reasonable numbers
            check_hessians_sparsity(); // here m_hs_dim is initialized
        }

        /// Copy constructor
        problem(const problem &other):
            m_ptr(other.m_ptr->clone()),
            m_fevals(0u),
            m_gevals(0u),
            m_hevals(0u),
            m_n(other.m_n),
            m_gs_dim(other.m_gs_dim),
            m_hs_dim(other.m_hs_dim)
        {}

        /// Move constructor
        problem(problem &&other) noexcept :
            m_ptr(std::move(other.m_ptr)),
            m_fevals(other.m_fevals.load()),
            m_gevals(other.m_gevals.load()),
            m_hevals(other.m_hevals.load()),
            m_n(other.m_n),
            m_gs_dim(other.m_gs_dim),
            m_hs_dim(other.m_hs_dim)
        {}

        /// Move assignment operator
        problem &operator=(problem &&other) noexcept
        {
            if (this != &other) {
                m_ptr = std::move(other.m_ptr);
                m_fevals.store(other.m_fevals.load());
                m_gevals.store(other.m_gevals.load());
                m_hevals.store(other.m_hevals.load());
                m_n = other.m_n;
                m_gs_dim = other.m_gs_dim;
                m_hs_dim = other.m_hs_dim;
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
         * @param[in] T The type of the orignal user-defined problem
         *
         * @return a const pointer to the user-defined problem
         * 
         */
        template <typename T>
        const T *extract() const
        {
            auto ptr = dynamic_cast<const detail::prob_inner<T> *>(m_ptr.get());
            if (ptr == nullptr) {
                return nullptr;
            }
            return &(ptr->m_value);
        }

        /// Checks the user defined problem type at run-time
        /**
         *
         * @param[in] T The type to be checked
         *
         * @return true if the user defined problem is \p T. false othewise.
         * 
         */
        template <typename T>
        bool is() const
        {
            return extract<T>();
        }

        /// Computes the fitness
        vector_double fitness(const vector_double &dv) const
        {
            // 1 - checks the decision vector
            check_decision_vector(dv);
            // 2 - computes the fitness
            vector_double retval(m_ptr->fitness(dv));
            // 3 - checks the fitness vector
            check_fitness_vector(retval);
            // 4 - increments fitness evaluation counter
            ++m_fevals;
            return retval;
        }

        /// Computes the gradient
        vector_double gradient(const vector_double &dv) const
        {
            // 1 - checks the decision vector
            check_decision_vector(dv);
            // 2 - compute the gradients
            vector_double retval(m_ptr->gradient(dv));
            // 3 - checks the gradient vector
            check_gradient_vector(retval);
            // 4 - increments gradient evaluation counter
            ++m_gevals;
            return retval;
        }

        /// Check if the user-define problem implements a gradient
        bool has_gradient() const
        {
            return m_ptr->has_gradient();
        }

        /// Returns the gradient sparsity
        sparsity_pattern gradient_sparsity() const
        {
            return m_ptr->gradient_sparsity();
        }

        /// Check if the user-defined dproblem implements a gradient_sparsity
        bool has_gradient_sparsity() const
        {
            return m_ptr->has_gradient_sparsity();
        }

        /// Computes the Hessians
        std::vector<vector_double> hessians(const vector_double &dv) const
        {
            // 1 - checks the decision vector
            check_decision_vector(dv);
            // 2 - computes the hessians
            std::vector<vector_double> retval(m_ptr->hessians(dv));
            // 3 - checks the hessians
            check_hessians_vector(retval);
            // 4 - increments hessians evaluation counter
            ++m_hevals;
            return retval;
        }

        /// Check if the user-defined dproblem implements the hessians
        bool has_hessians() const
        {
            return m_ptr->has_hessians();
        }

        /// Returns the hessians sparsity
        std::vector<sparsity_pattern> hessians_sparsity() const
        {
            return m_ptr->hessians_sparsity();
        }


        /// Check if the user-defined dproblem implements the hessians_sparosy
        bool has_hessians_sparsity() const
        {
            return m_ptr->has_hessians_sparsity();
        }

        /// Fitness dimension
        vector_double::size_type get_nobj() const
        {
            return m_ptr->get_nobj();
        }

        /// Decision vector dimension
        vector_double::size_type get_n() const
        {
            return m_n;
        }

        /// Box-bounds
        std::pair<vector_double, vector_double> get_bounds() const
        {
            return m_ptr->get_bounds();
        }

        /// Number of equality constraints
        vector_double::size_type get_nec() const
        {
            return m_ptr->get_nec();
        }

        /// Number of inequality constraints
        vector_double::size_type get_nic() const
        {
            return m_ptr->get_nic();
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

        /// Problem's name.
        std::string get_name() const
        {
            return m_ptr->get_name();
        }

        /// Extra info
        std::string get_extra_info() const
        {
            return m_ptr->get_extra_info();
        }

        /// human readable representation
        std::string human_readable() const
        {
            std::ostringstream s;
            s << "Problem name: " << get_name() << '\n';
            s << "\tGlobal dimension:\t\t\t" << get_n() << '\n';
            s << "\tFitness dimension:\t\t\t" << get_nobj() << '\n';
            s << "\tEquality constraints dimension:\t\t" << get_nec() << '\n';
            s << "\tInequality constraints dimension:\t" << get_nic() << '\n';
            s << "\tLower bounds: ";
            stream(s, get_bounds().first, '\n');
            s << "\tUpper bounds: ";
            stream(s, get_bounds().second, '\n');
            stream(s, "\n\tHas gradient: ", has_gradient(), '\n');
            stream(s, "\tUser implemented gradient sparsity: ", has_gradient_sparsity(), '\n');
            if (has_gradient()) {
                stream(s, "\tExpected gradients: ", m_gs_dim, '\n');
            }
            stream(s, "\tHas hessians: ", has_hessians(), '\n');
            stream(s, "\tUser implemented hessians sparsity: ", has_hessians_sparsity(), '\n');
            if (has_hessians()) {
                stream(s, "\tExpected hessian components: ", m_hs_dim, '\n');
            }
            stream(s, "\n\tFunction evaluations: ", get_fevals(), '\n');
            if (has_gradient()) {
                stream(s, "\tGradient evaluations: ", get_gevals(), '\n');
            }
            if (has_hessians()) {
                stream(s, "\tHessians evaluations: ", get_hevals(), '\n');
            }

            const auto extra_str = get_extra_info();
            if (!extra_str.empty()) {
                stream(s, "\nExtra info:\n", extra_str, '\n');
            }
            return s.str();
        }

        template <typename Archive>
        void save(Archive &ar) const
        {
            ar(m_ptr,m_fevals.load(), m_gevals.load(), m_hevals.load(), m_n, m_gs_dim, m_hs_dim);
        }
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
            ar(m_n, m_gs_dim,m_hs_dim);
        }

    private:
        template <typename U>
        static bool all_unique(std::vector<U> x)
        {
            std::sort(x.begin(),x.end());
            auto it = std::unique(x.begin(),x.end());
            return it == x.end();
        }
        // The gradient sparsity patter is, at this point, either the user
        // defined one or the default (dense) one.
        void check_gradient_sparsity()
        {
            auto gs = gradient_sparsity();
            auto n = get_n();
            auto nf = get_nobj() + get_nec() + get_nic();
            // 1 - We check that the gradient sparsity pattern has
            // valid indexes.
            for (const auto &pair: gs) {
                if ((pair.first >= nf) or (pair.second >= n)) {
                    pagmo_throw(std::invalid_argument,"Invalid pair detected in the gradient sparsity pattern: (" + std::to_string(pair.first) + ", " + std::to_string(pair.second) + ")\nFitness dimension is: " + std::to_string(nf) + "\nDecision vector dimension is: " + std::to_string(n));
                }
            }
            // 1bis We check all pairs are unique
            if (!all_unique(gs)) {
                pagmo_throw(std::invalid_argument, "Multiple entries of the same index pair was detected in the gradient sparsity pattern");
            }

            // 2 - We store the dimensions of the gradient sparsity pattern
            // as we will check that the returned gradient has this dimension
            m_gs_dim = gs.size();
        }
        void check_hessians_sparsity()
        {
            auto hs = hessians_sparsity();
            // 1 - We check that a hessian sparsity is provided for each component
            // of the fitness
            auto nf = get_nobj() + get_nec() + get_nic();
            if (hs.size()!=nf) {
                pagmo_throw(std::invalid_argument,"Invalid dimension of the hessians_sparsity: " + std::to_string(hs.size()) + ", expected: " + std::to_string(nf));
            }
            // 2 - We check that all hessian sparsity patterns have
            // valid indexes.
            for (const auto &one_hs: hs) {
                check_hessian_sparsity(one_hs);
            }
            // 3 - We store the dimensions of the hessian sparsity patterns
            // for future quick checks.
            m_hs_dim.clear();
            for (const auto &one_hs: hs) {
                m_hs_dim.push_back(one_hs.size());
            }

        }
        void check_hessian_sparsity(const sparsity_pattern& hs)
        {
            auto n = get_n();
            // 1 - We check that the hessian sparsity pattern has
            // valid indexes. Assuming a lower triangular representation of
            // a symmetric matrix. Example, for a 4x4 dense symmetric
            // [(0,0), (1,0), (1,1), (2,0), (2,1), (2,2), (3,0), (3,1), (3,2), (3,3)]
            for (const auto &pair: hs) {
                if ((pair.first >= n) or (pair.second > pair.first)) {
                    pagmo_throw(std::invalid_argument,"Invalid pair detected in the hessians sparsity pattern: (" + std::to_string(pair.first) + ", " + std::to_string(pair.second) + ")\nDecision vector dimension is: " + std::to_string(n) + "\nNOTE: hessian is a symmetric matrix and PaGMO represents it as lower triangular: i.e (i,j) is not valid if j>i");
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
            if (dv.size()!=get_n()) {
                pagmo_throw(std::invalid_argument,"Length of decision vector is " + std::to_string(dv.size()) + ", should be " + std::to_string(get_n()));
            }
            // 2 - Here is where one could check if the decision vector
            // is in the bounds. At the moment not implemented
        }

        void check_fitness_vector(const vector_double &f) const
        {
            auto nf = get_nobj() + get_nec() + get_nic();
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
            for (decltype(hs.size()) i=0u; i<hs.size(); ++i)
            {
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
        // Problem dimension
        vector_double::size_type m_n;
        // Expected dimensions of the returned gradient (matching the sparsity pattern)
        vector_double::size_type m_gs_dim;
        // Expected dimensions of the returned hessians (matching the sparsity patterns)
        std::vector<vector_double::size_type> m_hs_dim;
};

// Streaming operator for the class pagmo::problem
std::ostream &operator<<(std::ostream &os, const problem &p)
{
    os << p.human_readable() << '\n';
    return os;
}

} // namespaces

#endif
