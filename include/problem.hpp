#ifndef PAGMO_PROBLEM_HPP
#define PAGMO_PROBLEM_HPP

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

#define PAGMO_REGISTER_PROBLEM(prob) CEREAL_REGISTER_TYPE_WITH_NAME(pagmo::detail::prob_inner<prob>,#prob);

namespace pagmo
{

namespace detail
{

struct prob_inner_base
{
    virtual ~prob_inner_base() {}
    virtual prob_inner_base *clone() const = 0;
    virtual fitness_vector fitness(const decision_vector &) const = 0;
    virtual gradient_vector gradient(const decision_vector &) const = 0;
    virtual bool has_gradient() const = 0;
    virtual fitness_vector::size_type get_nf() const = 0;
    virtual decision_vector::size_type get_n() const = 0;
    virtual std::pair<decision_vector,decision_vector> get_bounds() const = 0;
    virtual decision_vector::size_type get_nec() const = 0;
    virtual decision_vector::size_type get_nic() const = 0;
    virtual std::string get_name() const = 0;
    virtual std::string extra_info() const = 0;
    template <typename Archive>
    void serialize(Archive &ar)
    {
        ar(m_sparsity);
    }
    sparsity_pattern m_sparsity;
};

template <typename T>
struct prob_inner: prob_inner_base
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
        "A problem must provide a fitness function and a method to query the number of objectives.");
    static_assert(has_dimensions_bounds<T>::value,
        "A problem must provide getters for its dimension and bounds.");
    prob_inner() = default;
    explicit prob_inner(T &&x):m_value(std::move(x))
    {
        set_sparsity();
    }
    explicit prob_inner(const T &x):m_value(x)
    {
        set_sparsity();
    }
    virtual prob_inner_base *clone() const override final
    {
        return ::new prob_inner<T>(m_value);
    }
    // Mandatory methods.
    virtual fitness_vector fitness(const decision_vector &dv) const override final
    {
        return m_value.fitness(dv);
    }
    virtual fitness_vector::size_type get_nf() const override final
    {
        return m_value.get_nf();
    }
    virtual decision_vector::size_type get_n() const override final
    {
        return m_value.get_n();
    }
    virtual std::pair<decision_vector,decision_vector> get_bounds() const override final
    {
        return m_value.get_bounds();
    }
    // Optional methods.
    template <typename U, typename std::enable_if<pagmo::has_gradient<U>::value,int>::type = 0>
    static gradient_vector gradient_impl(U &value, const decision_vector &dv)
    {
        return value.gradient(dv);
    }
    template <typename U, typename std::enable_if<!pagmo::has_gradient<U>::value,int>::type = 0>
    static gradient_vector gradient_impl(U &, const decision_vector &)
    {
        pagmo_throw(std::logic_error,"Gradients have been requested but not implemented.\nA function with prototype gradient_vector gradient(const decision_vector &x) was expected.");
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
    template <typename U, typename std::enable_if<has_sparsity<U>::value,int>::type = 0>
    void set_sparsity_impl(const U &value)
    {
        m_sparsity = value.sparsity();
    }
    template <typename U, typename std::enable_if<!has_sparsity<U>::value,int>::type = 0>
    void set_sparsity_impl(const U &)
    {
        // By default a problem is fully sparse
        auto dim = get_n(); 
        auto f_dim = get_nf() + get_nec() + get_nic();        
        sparsity_pattern retval;
        for (decltype(f_dim) j = 0u; j<f_dim; ++j) {
            for (decltype(dim) i = 0u; i<dim; ++i) {
               m_sparsity.push_back(std::pair<long, long>(j, i));
            }
        }
    }
    template <typename U, typename std::enable_if<has_constraints<U>::value,int>::type = 0>
    static decision_vector::size_type get_nec_impl(const U &value)
    {
        return value.get_nec();
    }
    template <typename U, typename std::enable_if<!has_constraints<U>::value,int>::type = 0>
    static decision_vector::size_type get_nec_impl(const U &)
    {
        return 0;
    }
    template <typename U, typename std::enable_if<has_constraints<U>::value,int>::type = 0>
    static decision_vector::size_type get_nic_impl(const U &value)
    {
        return value.get_nic();
    }
    template <typename U, typename std::enable_if<!has_constraints<U>::value,int>::type = 0>
    static decision_vector::size_type get_nic_impl(const U &)
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
    static std::string extra_info_impl(const U &value)
    {
        return value.extra_info();
    }
    template <typename U, typename std::enable_if<!has_extra_info<U>::value,int>::type = 0>
    static std::string extra_info_impl(const U &)
    {
        return "";
    }
    virtual gradient_vector gradient(const decision_vector &dv) const override final
    {
        return gradient_impl(m_value, dv);
    }
    virtual bool has_gradient() const override final
    {
        return has_gradient_impl(m_value);
    }
    void set_sparsity()
    {
        return set_sparsity_impl(m_value);
    }
    virtual decision_vector::size_type get_nec() const override final
    {
        return get_nec_impl(m_value);
    }
    virtual decision_vector::size_type get_nic() const override final
    {
        return get_nic_impl(m_value);
    }
    virtual std::string get_name() const override final
    {
        return get_name_impl(m_value);
    }
    virtual std::string extra_info() const override final
    {
        return extra_info_impl(m_value);
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

class problem
{
        // Enable the generic ctor only if T is not a problem (after removing
        // const/reference qualifiers).
        template <typename T>
        using generic_ctor_enabler = std::enable_if_t<!std::is_same<problem,std::decay_t<T>>::value,int>;
    public:
        template <typename T, generic_ctor_enabler<T> = 0>
        explicit problem(T &&x):m_ptr(::new detail::prob_inner<std::decay_t<T>>(std::forward<T>(x))),m_fevals(0u)
        {
            // check bounds consistency
            auto bounds = get_bounds();
            const auto &lb = bounds.first;
            const auto &ub = bounds.second;
            // 1 - check lower bounds length
            if (lb.size()!=get_n()) {
                pagmo_throw(std::invalid_argument,"Length of lower bounds vector is " + std::to_string(lb.size()) + ", should be " + std::to_string(get_n()));
            }
            // 2 - check upper bounds length
            if (ub.size()!=get_n()) {
                pagmo_throw(std::invalid_argument,"Length of upper bounds vector is " + std::to_string(ub.size()) + ", should be " + std::to_string(get_n()));
            }
            // 3 - checks lower < upper for all values in lb, lb
            for (decltype(lb.size()) i=0u; i < lb.size(); ++i) {
                if (lb[i] > ub[i]) {
                    pagmo_throw(std::invalid_argument,"The lower bound at position " + std::to_string(i) + " is " + std::to_string(lb[i]) +
                        "while the upper bound has the smaller value" + std::to_string(ub[i]) + "");
                }
            }
        }
        problem(const problem &other):m_ptr(other.m_ptr->clone()),m_fevals(0u) {}
        problem(problem &&other):m_ptr(std::move(other.m_ptr)),m_fevals(other.m_fevals.load()) {}

        template <typename T>
        const T *extract() const
        {
            auto ptr = dynamic_cast<const detail::prob_inner<T> *>(m_ptr.get());
            if (ptr == nullptr) {
                return nullptr;
            }
            return &(ptr->m_value);
        }

        template <typename T>
        bool is() const
        {
            return extract<T>();
        }

        fitness_vector fitness(const decision_vector &dv)
        {
            // 1 - checks the decision vector
            check_decision_vector(dv);
            // 2 - computes the fitness
            fitness_vector retval(m_ptr->fitness(dv));
            // 3 - checks the decision vector
            check_fitness_vector(retval);
            // 4 - increments fevals
            ++m_fevals;
            return retval;
        }
        gradient_vector gradient(const decision_vector &dv) const
        {
            // 1 - checks the decision vector
            check_decision_vector(dv);
            // 2 - compute the gradients
            gradient_vector retval(m_ptr->gradient(dv));
            // 3 - checks the decision vector
            check_gradient_vector(retval);
            return retval;
        }
        bool has_gradient() const
        {
            return m_ptr->has_gradient();
        } 
        const sparsity_pattern& sparsity() const
        {
            return m_ptr->m_sparsity;
        } 
        fitness_vector::size_type get_nf() const
        {
            return m_ptr->get_nf();
        }
        decision_vector::size_type get_n() const
        {
            return m_ptr->get_n();
        }
        std::pair<decision_vector, decision_vector> get_bounds() const
        {
            return m_ptr->get_bounds();
        }
        decision_vector::size_type get_nec() const
        {
            return m_ptr->get_nec();
        }
        decision_vector::size_type get_nic() const
        {
            return m_ptr->get_nic();
        }
       
        template <typename Archive>
        void serialize(Archive &ar)
        { 
            ar(m_ptr,m_fevals);
        }

        unsigned long long get_fevals() const
        {
            return m_fevals.load();
        }

        /// Get problem's name.
        /**
         * Will return the name of the user implemented problem as
         * returned by the detail::prob_inner method get_name
         *
         * @return name of the user implemented problem.
         */
        std::string get_name() const
        {
            return m_ptr->get_name();
        }

        std::string extra_info() const
        {
            return m_ptr->extra_info();
        } 

        std::string human_readable() const
        {
            std::ostringstream s;
            s << "Problem name: " << get_name() << '\n';
            //const size_type size = get_dimension();
            s << "\tGlobal dimension:\t\t\t" << get_n() << '\n';
            s << "\tFitness dimension:\t\t\t" << get_nf() << '\n';
            s << "\tEquality constraints dimension:\t\t" << get_nec() << '\n';
            s << "\tInequality constraints dimension:\t" << get_nic() << '\n';
            s << "\tLower bounds: ";
            stream(s, get_bounds().first, '\n');
            s << "\tUpper bounds: ";
            stream(s, get_bounds().second, '\n');
            const auto extra_str = extra_info();
            if (!extra_str.empty()) {
                stream(s, "\nExtra info:\n", extra_str, '\n');
            }
            return s.str();
        }

    private:
        void check_decision_vector(const decision_vector &dv) const
        {
            // 1 - check decision vector for length consistency
            if (dv.size()!=get_n()) {
                pagmo_throw(std::invalid_argument,"Length of decision vector is " + std::to_string(dv.size()) + ", should be " + std::to_string(get_n()));
            }
            // 2 - Here is where one could check if the decision vector
            // is in the bounds. At the moment not implemented
        }

        void check_fitness_vector(const fitness_vector &f) const
        {
            // Checks dimension of returned fitness
            if (f.size()!=get_nf()) {
                pagmo_throw(std::invalid_argument,"Fitness length is: " + std::to_string(f.size()) + ", should be " + std::to_string(get_nf()));
            }
        }

        void check_gradient_vector(const gradient_vector &gr) const
        {
            // Checks that the gradient vector returned has the same dimensions of the sparsity_pattern
            if (gr.size()!=m_ptr->m_sparsity.size()) {
                pagmo_throw(std::invalid_argument,"Gradients returned: " + std::to_string(gr.size()) + ", should be " + std::to_string(m_ptr->m_sparsity.size()));
            }
        }

    private:
        std::unique_ptr<detail::prob_inner_base> m_ptr;
        std::atomic<unsigned long long> m_fevals;
};

// Streaming operator for the class pagmo::problem
std::ostream &operator<<(std::ostream &os, const problem &p)
{
    os << p.human_readable() << '\n';
    return os;
}

} // namespaces

#endif
