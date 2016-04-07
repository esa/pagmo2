#ifndef PAGMO_PROBLEM_HPP
#define PAGMO_PROBLEM_HPP

#include <boost/lexical_cast.hpp>
#include <memory>
#include <type_traits>
#include <typeinfo>
#include <utility>

#include "exceptions.hpp"
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
    virtual fitness_vector fitness(const decision_vector &) = 0;
    virtual fitness_vector::size_type get_nf() const = 0;
    virtual decision_vector::size_type get_n() const = 0;
    virtual std::pair<decision_vector,decision_vector> get_bounds() const = 0;
    virtual decision_vector::size_type get_nec() const = 0;
    virtual decision_vector::size_type get_nic() const = 0;
    template <typename Archive>
    void serialize(Archive &) {}
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
    explicit prob_inner(T &&x):m_value(std::move(x)) {}
    explicit prob_inner(const T &x):m_value(x) {}
    virtual prob_inner_base *clone() const override
    {
        return ::new prob_inner<T>(m_value);
    }
    // Main methods.
    virtual fitness_vector fitness(const decision_vector &dv) override
    {
        return m_value.fitness(dv);
    }
    virtual fitness_vector::size_type get_nf() const override
    {
        return m_value.get_nf();
    }
    virtual decision_vector::size_type get_n() const override
    {
        return m_value.get_n();
    }
    virtual std::pair<decision_vector,decision_vector> get_bounds() const override
    {
        return m_value.get_bounds();
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
    virtual decision_vector::size_type get_nec() const override
    {
        return get_nec_impl(m_value);
    }
    virtual decision_vector::size_type get_nic() const override
    {
        return get_nic_impl(m_value);
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
    public:
        template <typename T>
        explicit problem(T &&x):m_ptr(::new detail::prob_inner<std::decay_t<T>>(std::forward<T>(x)))
        {
            // check bounds consistency
        }
        problem(const problem &other):m_ptr(other.m_ptr->clone()) {}
        problem(problem &&other) = default;

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
            // 1 - check decision vector for length consistency
            if (dv.size()!=get_n()) {
                pagmo_throw(std::invalid_argument,"Length of decision vector is " + boost::lexical_cast<std::string>(dv.size()) + ", should be " + boost::lexical_cast<std::string>(get_n()));
            }
            // 2 - Here is where one could check if the decision vector
            // is in the bounds. At the moment not implemented

            // 3 - computes the fitness
            fitness_vector retval(m_ptr->fitness(dv));
            // 4 - checks dimension of returned fitness
            if (retval.size()!=get_nf()) {
                pagmo_throw(std::invalid_argument,"Returned fitness length is: " + boost::lexical_cast<std::string>(retval.size()) + ", should be " + boost::lexical_cast<std::string>(get_nf()));
            }
            // 3 - increments m_fevals
            return retval;
        }
        fitness_vector::size_type get_nf() const
        {
            return m_ptr->get_nf();
        }
        decision_vector::size_type get_n() const
        {
            return m_ptr->get_n();
        }
        box_bounds get_bounds() const
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
            ar(m_ptr); 
        }
    private:
        std::unique_ptr<detail::prob_inner_base> m_ptr;
};

}

#endif
