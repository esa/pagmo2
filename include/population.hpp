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
#include "utils/constrained.hpp"

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

        const problem &get_problem() const 
        {
            return m_prob;
        }

        /// Getter for the fitness vectors
        const std::vector<vector_double> &get_f() const
        {
            return m_f;
        }

        /// Getter for the decision vectors
        const std::vector<vector_double> &get_x() const
        {
            return m_x;
        }

        /// Getter for the individual IDs
        std::vector<unsigned long long> get_ID() const
        {
            return m_ID;
        }

        /// Getter for the seed of the population random engine
        unsigned int get_seed() const
        {
            return m_seed;
        }

        /// Population champion
        vector_double::size_type champion() const
        {
            if (m_prob.get_nobj() > 1) {
                pagmo_throw(std::invalid_argument, "Champion can only be extracted in single objective problems");
            } 
            if (m_prob.get_nc() > 0) { // should we also code a min_element_population_con?
                return sort_population_con(m_f, m_prob.get_nec())[0];
            }
            // Sort for single objective, unconstrained optimization
            std::vector<vector_double::size_type> indexes(size());
            std::iota(indexes.begin(), indexes.end(), vector_double::size_type(0u));
            auto idx = std::min_element(indexes.begin(), indexes.end(), [this](auto idx1, auto idx2) {return m_f[idx1] < m_f[idx2];});
            return std::distance(indexes.begin(), idx);
        }

        /// Number of individuals in the population
        size_type size() const
        {
            return m_ID.size();
        }

        /// Serialization.
        template <typename Archive>
        void serialize(Archive &ar)
        {
            ar(m_prob, m_ID, m_x, m_f, m_e, m_seed);
        }

        /// Streaming operator for the class pagmo::problem
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
