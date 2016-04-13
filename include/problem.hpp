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

#define PAGMO_REGISTER_PROBLEM(prob) CEREAL_REGISTER_TYPE_WITH_NAME(pagmo::detail::prob_inner<prob>,#prob);

namespace pagmo
{

namespace detail
{

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
    virtual vector_double::size_type get_n() const = 0;
    virtual std::pair<vector_double,vector_double> get_bounds() const = 0;
    virtual vector_double::size_type get_nec() const = 0;
    virtual vector_double::size_type get_nic() const = 0;
    virtual std::string get_name() const = 0;
    virtual std::string get_extra_info() const = 0;
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
    virtual vector_double::size_type get_n() const override final
    {
        return m_value.get_n();
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
        pagmo_throw(std::logic_error,"Gradients have been requested but not implemented.\nA function with prototype vector_double gradient(const vector_double &x) const was expected.");
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
        auto dim = get_n(); 
        auto f_dim = get_nobj() + get_nec() + get_nic();        
        sparsity_pattern retval;
        for (decltype(f_dim) j = 0u; j<f_dim; ++j) {
            for (decltype(dim) i = 0u; i<dim; ++i) {
               retval.push_back({j, i});
            }
        }
        return retval;
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
        pagmo_throw(std::logic_error,"Hessians have been requested but not implemented.\nA function with prototype std::vector<vector_double> hessians(const vector_double &x) const was expected.");
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
        auto dim = get_n(); 
        auto f_dim = get_nobj() + get_nec() + get_nic();        
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
    template <typename U, typename std::enable_if<has_constraints<U>::value,int>::type = 0>
    static vector_double::size_type get_nec_impl(const U &value)
    {
        return value.get_nec();
    }
    template <typename U, typename std::enable_if<!has_constraints<U>::value,int>::type = 0>
    static vector_double::size_type get_nec_impl(const U &)
    {
        return 0;
    }
    template <typename U, typename std::enable_if<has_constraints<U>::value,int>::type = 0>
    static vector_double::size_type get_nic_impl(const U &value)
    {
        return value.get_nic();
    }
    template <typename U, typename std::enable_if<!has_constraints<U>::value,int>::type = 0>
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

class problem
{
        // Enable the generic ctor only if T is not a problem (after removing
        // const/reference qualifiers).
        template <typename T>
        using generic_ctor_enabler = std::enable_if_t<!std::is_same<problem,std::decay_t<T>>::value,int>;
    public:
        template <typename T, generic_ctor_enabler<T> = 0>
        explicit problem(T &&x):m_ptr(::new detail::prob_inner<std::decay_t<T>>(std::forward<T>(x))),m_fevals(0u),m_gevals(0u),m_hevals(0u)
        {
            // check bounds consistency
            auto bounds = get_bounds();
            const auto &lb = bounds.first;
            const auto &ub = bounds.second;
            // 1 - check lower bounds length
            if (lb.size()!=get_n()) {
                pagmo_throw(std::invalid_argument,"Length of lower bounds vector is " + std::to_string(lb.size()) + ", expected " + std::to_string(get_n()));
            }
            // 2 - check upper bounds length
            if (ub.size()!=get_n()) {
                pagmo_throw(std::invalid_argument,"Length of upper bounds vector is " + std::to_string(ub.size()) + ", expected " + std::to_string(get_n()));
            }
            // 3 - checks lower < upper for all values in lb, lb
            for (decltype(lb.size()) i=0u; i < lb.size(); ++i) {
                if (lb[i] > ub[i]) {
                    pagmo_throw(std::invalid_argument,"The lower bound at position " + std::to_string(i) + " is " + std::to_string(lb[i]) +
                        " while the upper bound has the smaller value " + std::to_string(ub[i]));
                }
            }
            // 4 - checks that the sparsity contains reasonable numbers
            check_gradient_sparsity(); // here m_gs_dim is initialized
            // 5 - checks that the hessians contain reasonable numbers
            check_hessians_sparsity(); // here m_hs_dim is initialized
        }
        problem(const problem &other):
            m_ptr(other.m_ptr->clone()),
            m_fevals(0u),
            m_gevals(0u),
            m_hevals(0u),
            m_gs_dim(other.m_gs_dim),
            m_hs_dim(other.m_hs_dim)
        {}
        problem(problem &&other) noexcept :
            m_ptr(std::move(other.m_ptr)),
            m_fevals(other.m_fevals.load()),
            m_gevals(other.m_gevals.load()),
            m_hevals(other.m_hevals.load()),
            m_gs_dim(other.m_gs_dim),
            m_hs_dim(other.m_hs_dim)
        {}

        problem &operator=(problem &&other) noexcept
        {
            if (this != &other) {
                m_ptr = std::move(other.m_ptr);
                m_fevals.store(other.m_fevals.load());
                m_gevals.store(other.m_gevals.load());
                m_hevals.store(other.m_hevals.load());
                m_gs_dim = other.m_gs_dim;
                m_hs_dim = other.m_hs_dim;
            }
            return *this;
        }
        problem &operator=(const problem &other)
        {
            // Copy ctor + move assignment.
            return *this = problem(other);
        }

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

        vector_double fitness(const vector_double &dv)
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
        vector_double gradient(const vector_double &dv)
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
        bool has_gradient() const
        {
            return m_ptr->has_gradient();
        }
        std::vector<vector_double> hessians(const vector_double &dv)
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
        bool has_hessians() const
        {
            return m_ptr->has_hessians();
        }
        sparsity_pattern gradient_sparsity() const
        {
            return m_ptr->gradient_sparsity();
        }
        bool has_gradient_sparsity() const
        {
            return m_ptr->has_gradient_sparsity();
        }
        std::vector<sparsity_pattern> hessians_sparsity() const
        {
            return m_ptr->hessians_sparsity();
        } 
        bool has_hessians_sparsity() const
        {
            return m_ptr->has_hessians_sparsity();
        }
        vector_double::size_type get_nobj() const
        {
            return m_ptr->get_nobj();
        }
        vector_double::size_type get_n() const
        {
            return m_ptr->get_n();
        }
        std::pair<vector_double, vector_double> get_bounds() const
        {
            return m_ptr->get_bounds();
        }
        vector_double::size_type get_nec() const
        {
            return m_ptr->get_nec();
        }
        vector_double::size_type get_nic() const
        {
            return m_ptr->get_nic();
        }       
        unsigned long long get_fevals() const
        {
            return m_fevals.load();
        }
        unsigned long long get_gevals() const
        {
            return m_gevals.load();
        }
        unsigned long long get_hevals() const
        {
            return m_hevals.load();
        }
        vector_double::size_type get_gs_dim() const 
        {
            return m_gs_dim;
        }
        std::vector<vector_double::size_type> get_hs_dim() const
        {
            return m_hs_dim;
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

        std::string get_extra_info() const
        {
            return m_ptr->get_extra_info();
        } 

        std::string human_readable() const
        {
            std::ostringstream s;
            s << "Problem name: " << get_name() << '\n';
            //const size_type size = get_dimension();
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
            ar(m_ptr,m_fevals.load(), m_gevals.load(), m_hevals.load(), m_gs_dim, m_hs_dim);
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
            ar(m_gs_dim,m_hs_dim);
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
            for (auto pair: gs) {
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
            for (auto one_hs: hs) {
                check_hessian_sparsity(one_hs);
            }
            // 3 - We store the dimensions of the hessian sparsity patterns
            // for future quick checks.
            m_hs_dim.clear();
            for (auto one_hs: hs) {
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
            for (auto pair: hs) {
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
        std::atomic<unsigned long long> m_fevals;
        // Atomic counter for calls to the gradient 
        std::atomic<unsigned long long> m_gevals;
        // Atomic counter for calls to the hessians 
        std::atomic<unsigned long long> m_hevals;
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
