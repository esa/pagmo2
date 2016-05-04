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
#include "utils/generic.hpp"
#include "utils/constrained.hpp"

namespace pagmo
{
/// Population class.
/**
 * \image html population.jpg
 *
 * This class represents a population of individuals, i.e. potential
 * candidate solutions to a given problem. In PaGMO an
 * individual is determined 
 * - by a unique ID used to track him across generations and migrations
 * - by a chromosome (a decision vector)
 * - by the fitness of the chromosome as evaluated by a pagmo::problem.
 * and thus including objectives, equality constraints and inequality
 * constraints if present.
 * 
 */
class population
{
    private:
    // A shortcut to std::vector<vector_double>::size_type
    using size_type = std::vector<vector_double>::size_type;

    public:
        /// Default constructor
        /**
         * Constructs an empty population with a pagmo::null_problem
         */
        population() : m_prob(null_problem{}), m_ID(), m_x(), m_f(), m_e(0u), m_seed(0u) {}

        /// Constructor
        /**
         * Constructs a population with \p pop_size individuals associated
         * to the pagmo::problem p and setting the population random seed
         * to \p seed
         *
         * @param[in] p the pagmo::problem the population refers to
         * @param[in] pop_size population size (i.e. number of individuals therein)
         * @param[in] seed seed of the random number generator used, for example, to
         * create new random individuals within the bounds
         *
         * @throws unspecified any excpetion thrown by random_decision_vector()
         *
         */
        explicit population(const pagmo::problem &p, size_type pop_size = 0u, unsigned int seed = pagmo::random_device::next()) : m_prob(p), m_e(seed), m_seed(seed)
        {
            for (decltype(pop_size) i = 0u; i < pop_size; ++i) {
                push_back(decision_vector());
            }
        }

        /// Adds one decision vector (chromosome) to the population
        /** 
         * Appends a new chromosome \p x to the population, evaluating
         * its fitness and creating a new unique identifier for the newly
         * born individual
         */

        void push_back(const vector_double &x)
        {
            auto new_id = std::uniform_int_distribution<unsigned long long>()(m_e);
            m_ID.push_back(new_id);
            m_x.push_back(x);
            m_f.push_back(m_prob.fitness(x));
        }

        /// Creates a random decision vector 
        /**
         * Creates a random decision vector within the problem's bounds.
         * It calls internally pagmo::decision_vector
         *
         * @returns a random decision vector
         *
         * @throws unspecified all excpetions thrown by pagmo::decision_vector
         */
        vector_double decision_vector() const
        {
            return pagmo::decision_vector(m_prob.get_bounds(), std::uniform_int_distribution<unsigned int>()(m_e));
        }

        /// Sets the \f$i\f$-th individual's chromosome
        /**
         *
         * Sets the chromosome of the \f$i\f$-th individual to the 
         * value \p x and changes its fitness accordingly. The
         * individual's ID remains the same
         *
         * @note a call to this method triggers one fitness function evaluation
         *
         * @param[in] i individual's index in the population
         * @param[in] x decision vector
         *
         * @throws unspecified any exception thrown by set_xf
         */
        void set_x(size_type i, const vector_double &x)
        {
            set_xf(i, x, m_prob.fitness(x));
        }

        /// Sets the \f$i\f$-th individual decision vector, and fitness
        /**
         * Sets simultaneously the \f$i\f$-th individual decision vector
         * and fitness thus avoiding to trigger a fitness function evaluation
         * 
         * @note: The user must make sure that the input fitness \p f makes sense
         * as pagmo will only check its dimension.
         *
         * @param[in] i individual's index in the population
         * @param[in] x a decision vector (chromosome)
         * @param[in] f a fitness vector
         *
         */
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

        /// Getter for the pagmo::problem
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
        const std::vector<unsigned long long> &get_ID() const
        {
            return m_ID;
        }

        /// Getter for the seed of the population random engine
        unsigned int get_seed() const
        {
            return m_seed;
        }

        /// Population champion
        /**
         * The best individual of a population is defined as its *champion*.
         * If the problem is single-objective and unconstrained ,the champion
         * is simply the individual with the smallest fitness. If the problem 
         * is, instead, single objective, but with constraints, the best individual
         * will be defined using the criteria specified in pagmo::sort_population_con.
         * If the problem is multi-objective one single champion is not defined. In
         * this case the user can still obtain a strict ordering of the population
         * individuals by calling the pagmo::sort_population_mo function
         */
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
            return static_cast<vector_double::size_type>(std::distance(indexes.begin(), idx));
        }

        /// Number of individuals in the population
        size_type size() const
        {
            return m_ID.size();
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
        friend class cereal::access; 
        // Serialization.
        template <typename Archive>
        void serialize(Archive &ar)
        {
            ar(m_prob, m_ID, m_x, m_f, m_e, m_seed);
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
