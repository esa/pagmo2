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
 * @see http://arxiv.org/pdf/cond-mat/9803344 for the paper inroducing the basin hopping idea for a Lennard-Jones cluster optimization
 */
class mbh : public algorithm
{
public:
    /// Default constructor, only here as serialization requires it
    mbh() : algorithm(compass_search{}), m_stop(5u), m_perturb(1, 1e-2), m_e(0u) ,m_seed(0u)
    {
    }
    /// Constructor
    template <typename T>
    explicit mbh(T &&a, unsigned int stop, double perturb, unsigned int seed = pagmo::random_device::next())
     : algorithm(std::forward<T>(a)), m_stop(stop), m_perturb(1, perturb), m_e(seed) ,m_seed(seed)
    {
    }
    /// Algorithm evolve method (juice implementation of the algorithm)
    population evolve(population pop) const
    {
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
    //population evolve(const population &pop) const = delete;
    //template <typename Archive>
    //void serialize(Archive &ar) = delete;
    bool has_set_seed() const = delete;
    bool is_stochastic() const = delete;
    void set_verbosity(unsigned int level) = delete;
    bool has_set_verbosity() const = delete;


    unsigned int m_stop;
    // This is mutable as to allow to construct mbh from a perturbation defined as a scalar
    // (in which case upon the first call to evolve it is expanded to the problem dimension)
    // and as a vector (in which case mbh will only operate on problem having the correct dimension)
    mutable std::vector<double> m_perturb;
    mutable detail::random_engine_type m_e;
    unsigned int m_seed;
};
}
PAGMO_REGISTER_ALGORITHM(pagmo::mbh)

#endif
