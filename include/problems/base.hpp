#ifndef PAGMO_PROBLEMS_BASE_HPP
#define PAGMO_PROBLEMS_BASE_HPP

#include "../problem.hpp"

namespace pagmo
{

namespace problems
{

using decision_vector = std::vector<double>;
using decision_vector_int = std::vector<long long>;
using constraint_vector = std::vector<double>;
using fitness_vector = std::vector<double>;

class base
{
    /// Problem continuous dimension
    using size_type = decision_vector::size_type;
    /// Problem integer dimension
    using i_size_type = decision_vector_int::size_type;
    /// Problem constraint dimension
    using c_size_type = constraint_vector::size_type;
    /// Problem fitness dimension
    using f_size_type = fitness_vector::size_type;

    public:
        base() = default;
        explicit base(const decision_vector &lb, 
             const decision_vector &ub, 
             unsigned int n = 0, 
             unsigned int ni = 1, 
             unsigned int nf = 0, 
             unsigned int nc = 0, 
             unsigned int nic = 0, 
             const constraint_vector &c_tol = constraint_vector()) : m_lb(lb), m_ub(ub), m_dimension(n), m_i_dimension(ni), m_f_dimension(nf), m_c_dimension(nc), m_ic_dimension(nic), m_c_tol(c_tol)
        {
            // sanity checks
        }
        base(base &&) = default;
        base(const base &) = default;

        virtual fitness_vector objfun(const decision_vector &x, const decision_vector_int &x_i) const
        {
            fitness_vector retval(m_f_dimension, 0.);
            return retval;
        }

        virtual constraint_vector constraints(const decision_vector &x, const decision_vector_int &x_i) const
        {
            constraint_vector retval(m_c_dimension, 0.);
            return retval;
        }

        template <typename Archive>
        void serialize(Archive &ar)
        {
            ar(m_lb);
            ar(m_ub);
            ar(m_dimension);
            ar(m_i_dimension);
            ar(m_f_dimension);
            ar(m_c_dimension);
            ar(m_ic_dimension);
            ar(m_c_tol);
            ar(m_best_x);
            ar(m_best_f);
            ar(m_best_c);
            ar(m_fevals);
            ar(m_cevals);
        }
        
    private:
        // Data members.
        // Lower bounds.
        decision_vector                 m_lb;
        // Upper bounds.
        decision_vector                 m_ub;
        // Size of the continuous part of the problem.
        const size_type                 m_dimension = 1;
        // Size of the integer part of the problem.
        const size_type                 m_i_dimension = 0;
        // Size of the fitness vector.
        const f_size_type               m_f_dimension = 1;
        // Global constraints dimension.
        const c_size_type               m_c_dimension = 0;
        // Inequality constraints dimension
        const c_size_type               m_ic_dimension = 0;
        // Tolerance for constraints violation.
        const std::vector<double>       m_c_tol;

        // Best known solutions (default to empty -> unknown)
        std::vector<decision_vector>    m_best_x = std::vector<decision_vector>();
        std::vector<fitness_vector>     m_best_f = std::vector<fitness_vector>();
        std::vector<constraint_vector>  m_best_c = std::vector<fitness_vector>();

        // Number of function and constraints evaluations
        unsigned int                    m_fevals = 0;
        unsigned int                    m_cevals = 0;
};

}} //namespaces

//PAGMO_REGISTER_PROBLEM(pagmo::problems::base);

#endif
