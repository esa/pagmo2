#ifndef PAGMO_ALGORITHMS_MBH_HPP
#define PAGMO_ALGORITHMS_MBH_HPP

#include <iomanip>
#include <random>
#include <string>
#include <tuple>

#include "../algorithm.hpp"
#include "../exceptions.hpp"
#include "../io.hpp"
#include "../population.hpp"
#include "../rng.hpp"

namespace pagmo
{

/// Monotonic Basin Hopping (generalized)
/**
 * \image html mbh.png "A schematic diagram illustrating the fitness landscape as seen by basin hopping" width=3cm
 *
 * Monotonic basin hopping, or simply, basin hopping, is an algorithm rooted in the idea of mapping
 * the objective function \f$f(\mathbf x_0)\f$ into the local minima found starting from \f$\mathbf x_0\f$.
 * This simple idea allows a substantial increase of efficiency in solving problems, such as the Lennard-Jones
 * cluster or the MGA-1DSM interplanetary trajectory problem that are conjectured to have a so-called
 * funnel structure.
 *
 * In pagmo we provide an original generalization of this concept resulting in a meta-algorithm that operates
 * on any pagmo::population using any suitable pagmo::algorithm. When a population containing a single
 * individual is used and coupled with a local optimizer, the original method is recovered.
 * The pseudo code of our generalized version is:
 * @code
 * > Select a pagmo::population
 * > Select a pagmo::algorithm
 * > Store best individual
 * > while i < stop_criteria
 * > > Perturb the population in a selected neighbourhood
 * > > Evolve the population using the algorithm
 * > > if the best individual is improved
 * > > > increment i
 * > > > update best individual
 * > > else
 * > > > i = 0
 * @endcode
 *
 * @see http://arxiv.org/pdf/cond-mat/9803344 for the paper inroducing the basin hopping idea for a Lennard-Jones
 * cluster optimization
 */
class mbh : public algorithm
{
public:
    /// Default constructor, only here as serialization requires it
    mbh() : algorithm(compass_search{}), m_stop(5u), m_perturb(1, 1e-2), m_e(0u), m_seed(0u)
    {
    }
    /// Constructor
    template <typename T>
    explicit mbh(T &&a, unsigned int stop, double perturb, unsigned int seed = pagmo::random_device::next())
        : algorithm(std::forward<T>(a)), m_stop(stop), m_perturb(1, perturb), m_e(seed), m_seed(seed)
    {
    }
    /// Algorithm evolve method (juice implementation of the algorithm)
    population evolve(population pop) const
    {
        // We store some useful variables
        const auto &prob = pop.get_problem(); // This is a const reference, so using set_seed for example will not be
                                              // allowed (pop.set_problem_seed is)
        auto dim = prob.get_nx();             // This getter does not return a const reference but a copy
        const auto bounds = prob.get_bounds();
        const auto &lb = bounds.first;
        const auto &ub = bounds.second;
        auto prob_f_dimension = prob.get_nf();

        auto fevals0 = prob.get_fevals(); // discount for the already made fevals
        unsigned int count = 1u;          // regulates the screen output

        // PREAMBLE-------------------------------------------------------------------------------------------------
        // No need for one as the inner algorithm will be tasked to check the population compatibility
        // ---------------------------------------------------------------------------------------------------------
        // Check if the perturbation vector has size 1, in which case the whole perturbation vector is filled with
        // m_perturb[0]
        if (m_perturb.size() == 1u) {
            for (decltype(dim) i=1u; i<dim; ++i) {
                m_perturb.push_back(m_perturb[0]);
            }
        }
        // Check that the perturbation vector size equals the size of the problem
        if (m_perturb.size() != dim) {
            pagmo_throw(std::invalid_argument, "The perturbation vector size is: " + std::to_string(m_perturb.size())
                                                   + ", while the problem dimension is: " + std::to_string(dim)
                                                   + ". They need to be equal for MBH to work.");
        }
        // Get out if there is nothing to do.
        if (m_stop == 0u) {
            return pop;
        }
        // We extract chromosomes and fitnesses
        return static_cast<const algorithm *>(this)->evolve(pop);
    }
    /// Sets the algorithm seed
    void set_seed(unsigned int seed)
    {
        m_seed = seed;
    }
    /// Gets the seed
    unsigned int get_seed() const
    {
        return m_seed;
    }
    /// Algorithm name
    std::string get_name() const
    {
        return "Monotonic Basin Hopping (MBH) - Generalized";
    }
    /// Extra informations
    std::string get_extra_info() const
    {
        std::ostringstream ss;
        stream(ss, "\tStop: ", m_stop);
        stream(ss, "\n\tPerturbation vector: ", m_perturb);
        stream(ss, "\n\tSeed: ", m_seed);
        stream(ss, "\n\n\tInner algorithm: ", static_cast<const algorithm *>(this)->get_name());
        stream(ss, "\n\tInner algorithm extra info: ");
        stream(ss, "\n", static_cast<const algorithm *>(this)->get_extra_info());
        return ss.str();
    }

    /// Serialization
    template <typename Archive>
    void serialize(Archive &ar)
    {
        ar(cereal::base_class<algorithm>(this), m_stop, m_perturb, m_e, m_seed);
    }

private:
    // Delete all that we do not want to inherit from problem
    // A - Common to all meta
    template <typename T>
    bool is() const = delete;
    // population evolve(const population &pop) const = delete;
    // template <typename Archive>
    // void serialize(Archive &ar) = delete;
    bool has_set_seed() const = delete;
    bool is_stochastic() const = delete;
    void set_verbosity(unsigned int level) = delete;
    bool has_set_verbosity() const = delete;
    // NOTE: We delete the streaming operator overload called with mbh, otherwise the inner algo would stream
    // NOTE: If a streaming operator is wanted for this class remove the line below and implement it
    friend std::ostream &operator<<(std::ostream &, const mbh &) = delete;

    unsigned int m_stop;
    // The member m_perturb is mutable as to allow to construct mbh also using a perturbation defined as a scalar
    // (in which case upon the first call to evolve it is expanded to the problem dimension)
    // and as a vector (in which case mbh will only operate on problem having the correct dimension)
    // While the use of "mutable" is not encouraged, in this case the alternative would be to have the user
    // construct the mbh algo passing one further parameter (the problem dmension) rather than having this determined
    // upon
    // the first call to evolve.
    mutable std::vector<double> m_perturb;
    mutable detail::random_engine_type m_e;
    unsigned int m_seed;
};
}
PAGMO_REGISTER_ALGORITHM(pagmo::mbh)

#endif
