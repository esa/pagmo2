#ifndef PAGMO_POPULATION_H
#define PAGMO_POPULATION_H

#include <cmath>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "problem.hpp"
#include "problems/null_problem.hpp"
#include "rng.hpp"
#include "serialization.hpp"
#include "types.hpp"
#include "utils/pareto.hpp"

namespace pagmo
{

class population
{
    private:
    // A shortcut to std::vector<vector_double>::size_type
    using size_type = std::vector<vector_double>::size_type;

    public:
        /// Default constructor
        population() : m_prob(null_problem{}), m_ID(), m_x(), m_f(), m_e(0u), m_seed(0u) {}

        /// Constructor
        explicit population(const pagmo::problem &p, size_type size = 0u, unsigned int seed = pagmo::random_device::next()) : m_prob(p), m_e(seed), m_seed(seed)
        {
            for (decltype(size) i = 0u; i < size; ++i) {
                push_back(random_decision_vector());
            }
        }

        // Appends a new decision vector to the population creating a unique
        // ID and comuting the fitness
        void push_back(const vector_double &x)
        {
            auto new_id = std::uniform_int_distribution<unsigned long long>()(m_e);
            m_ID.push_back(new_id);
            m_x.push_back(x);
            m_f.push_back(m_prob.fitness(x));
        }

        // Creates a random decision_vector within the problem bounds [lb, ub)
        vector_double random_decision_vector() const
        {
            const auto dim = m_prob.get_nx();
            const auto bounds = m_prob.get_bounds();
            // This will check for consistent vector lengths, lb <= ub and no NaNs.
            detail::check_problem_bounds(bounds);
            if (bounds.first.size() != dim) {
                pagmo_throw(std::invalid_argument,"Problem bounds are inconsistent with problem dimension");
            }
            vector_double retval(dim);
            for (decltype(m_prob.get_nx()) i = 0u; i < dim; ++i) {
                // NOTE: see here for the requirements for floating-point RNGS:
                // http://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution/uniform_real_distribution

                // 1 - Forbid random generation when bounds are infinite.
                if (std::isinf(bounds.first[i]) || std::isinf(bounds.second[i])) {
                    pagmo_throw(std::invalid_argument,"Cannot generate a random individual if the problem is"
                     " unbounded (inf bounds detected)");
                }
                // 2 - Bounds cannot be too large.
                const auto delta = bounds.second[i] - bounds.first[i];
                if (!std::isfinite(delta) || delta > std::numeric_limits<double>::max()) {
                    pagmo_throw(std::invalid_argument,"Cannot generate a random individual if the problem bounds "
                        "are too large");
                }
                // 3 - If the bounds are equal we don't call the RNG, as that would be undefined behaviour.
                if (bounds.first[i] == bounds.second[i]) {
                    retval[i] = bounds.first[i];
                } else {
                    retval[i] = std::uniform_real_distribution<double>(bounds.first[i], bounds.second[i])(m_e);
                }
            }
            return retval;
        }

        // Changes the i-th individual decision vector, causing a fitness evaluation
        // ID is unchanged
        void set_x(size_type i, const vector_double &x)
        {
            set_xf(i, x, m_prob.fitness(x));
        }

        // Changes the i-th individual decision vector, and fitness
        // ID is unchanged
        void set_xf(size_type i, const vector_double &x, const vector_double &f)
        {
            if (i >= size()) {
                pagmo_throw(std::invalid_argument,"Trying to access individual at position: " 
                    + std::to_string(i) 
                    + ", while population has size: " 
                    + std::to_string(size()));
            }
            if (f.size() != m_prob.get_nf()) {
                pagmo_throw(std::invalid_argument,"Trying to set a fitness of dimension: "  
                    + std::to_string(f.size()) 
                    + ", while problem get_nf returns: " 
                    + std::to_string(m_prob.get_nf())
                );
            }
            m_x[i] = x;
            m_f[i] = f;
        }

        const std::vector<vector_double> &get_f() const
        {
            return m_f;
        }

        const std::vector<vector_double> &get_x() const
        {
            return m_x;
        }

        std::vector<unsigned long long> get_ID() const
        {
            return m_ID;
        }

        // Gets the the seed of the population random engine
        unsigned int get_seed() const
        {
            return m_seed;
        }

        // Number of individuals in the population
        size_type size() const
        {
            return m_ID.size();
        }

        /// Fitness Comparison
        /**
         *  Default comparison between two fitness vectors \f$f_1\f$ and \f$f_2\f$.
         *  It returns true if \f$f_1\f$ is better than \f$f_2\f$, false otherwise,
         *  that is if:
         *   - \f$f_1\f$ is feasible and \f$f_2\f$ is not.
         *   - \f$f_1\f$ and \f$f_2\f$ are both infeasible, but the number
         *   of constraints violated by \f$f_1\f$ is less than the number of
         *   constraints violated by \f$f_2\f$. In case of a tie the smallest 
         *   \f$l_2\f$ norm of the constraint violation is used instead.
         *   - \f$f_1\f$ and \f$f_2\f$ are both feasible, but the non domination
         *  rank of \f$f_1\f$ is smaller than that of \f$f_2\f$. In case of a
         *   tie the largest crowding distance is used instead.
         *
         * This comparison defines a strict ordering over the individuals
         */
        struct fitness_comparison 
        {
            fitness_comparison(const vector_double &tol = vector_double{}) : m_tol(tol), m_tuple(), m_crowding() 
            {
                // If the tolerance vector is empty (default) set tolerances to zero
                if (m_tol.size()==0) {
                    m_tol.resize(m_prob.get_nic() + m_prob.get_nec());
                    std::fill(m_tol.begin(),m_tol.end(), 0.);
                }
                // Check that the tolerances size equals the size of the constraints
                if (m_tol.size()!=m_prob.get_nic() + m_prob.get_nec())
                {
                    pagmo_throw(std::invalid_argument, "The vector of constraint tolerances has dimension: " + std::to_string(m_tol.size()) + 
                        " while the problem constraint are " + std::string(get_nic()+get_nec()));
                }
                // Run fast-non-dominated sorting and crowding distance for the population
                m_tuple = fast_non_dominated_sorting(m_f);
                for (auto front: std::get<0>(m_tuple)) {
                    std::vector<vector_double> non_dom_fits(front.size());
                    for (auto i = 0u; i < front.size(); ++i) {
                        non_dom_fits[i] = m_fit(front[i]);
                    }
                    auto tmp = crowding_distance(non_dom_fits);
                    for (auto i = 0u; i < front.size(); ++i) {
                        m_crowding[front[i]] = tmp[i];
                    }
                }
            }

            bool operator() (vector_double::size_type idx1,vector_double::size_type idx2)
            { 
                // Shortcut for unconstrained single objective (do we need it?)
                if (m_tol.size()==0u && m_prob.get_nobj()==1u) {
                    return m_f[idx1][0] < m_f[idx2][0]; // the only objective decides.
                }

                auto feas1 = violations(idx1, m_tol); // returns a tuple with the number of constraint violated and the total L2 norm of the violation
                auto feas2 = violations(idx2, m_tol);
                if (std::get<0>(feas1) == std::get<0>(feas2)) { // same number of constraint violated
                    if (std::get<0>(feas1) > 0u) { // both unfeasible
                        return std::get<1>(feas1) < std::get<1>(feas2);
                    } else { // both feasible
                        if (m_prob.get_nobj() == 1) {
                            return m_f[idx1][0] < m_f[idx2][0];
                        } else {
                            if (std::get<3>(m_tuple)[idx1] == std::get<3>(m_tuple)[idx2]) { // same non domination rank
                                return m_crowding[idx1] > m_crowding[idx2]; // crowding distance decides
                            } else {
                                return std::get<3>(m_tuple)[idx1] < std::get<3>(m_tuple)[idx2]; // non domination rank decides
                            }
                        }
                    }
                } else { // both unfeasible with a different number of violations
                    return feas1 < feas2; // number of constraint violations decide
                }
            }

            vector_double m_tol;
            fnds_return_value m_tuple;
            vector_double m_crowding;
        };

        std::tuple<unsigned int, double> violations(vector_double::size_type idx, const vector_double &tol) 
        {
            return std::make_tuple(0,0);
        }

        // Serialization.
        template <typename Archive>
        void serialize(Archive &ar)
        {
            ar(m_prob, m_ID, m_x, m_f, m_e, m_seed);
        }

        // Streaming operator for the class pagmo::problem
        friend std::ostream &operator<<(std::ostream &os, const population &p)
        {
            stream(os, p.m_prob, '\n');
            stream(os, "Population size: ",p.size(),"\n\n");
            stream(os, "List of individuals: ",'\n');
            for (size_type i=0u; i < p.size(); ++i) {
                stream(os, "#", i, ":\n");
                stream(os, "\tID:\t\t\t", p.m_ID[i], '\n');
                stream(os, "\tDecision vector:\t", p.m_x[i], '\n');
                stream(os, "\tFitness vector:\t\t", p.m_f[i], '\n');
            }
            return os;
        }

    private:
        // Problem.
        problem                                m_prob;
        // ID of the various decision vectors
        std::vector<unsigned long long>        m_ID;
        // Decision vectors.
        std::vector<vector_double>             m_x;
        // Fitness vectors.
        std::vector<vector_double>             m_f;
        // Random engine.
        mutable detail::random_engine_type     m_e;
        // Seed.
        unsigned int                           m_seed;
};

} // namespace pagmo


#endif
