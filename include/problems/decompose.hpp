#ifndef PAGMO_PROBLEM_DECOMPOSE_HPP
#define PAGMO_PROBLEM_DECOMPOSE_HPP

#include <algorithm>
#include <cassert>
#include <limits>
#include <numeric>
#include <string>

#include "../exceptions.hpp"
#include "../problem.hpp"
#include "../serialization.hpp"
#include "../types.hpp"
#include "zdt.hpp"

namespace pagmo
{

    /// Decompose meta-problem
    /**
     * \image html decompose.png "Decomposition." width=3cm
     *
     * This meta-problem *decomposes* a multi-objective input problem (user-defined or a pagmo::problem),
     * resulting in a single-objective pagmo::problem with a fitness function combining the original fitness functions.
     * In particular, three different *decomposition methods* are here made available:
     *
     * - weighted decomposition
     * - Tchebycheff decomposition
     * - boundary interception method (with penalty constraint)
     *
     * In the case of \f$n\f$ objectives, we indicate with: \f$ \mathbf f(\mathbf x) = [f_1(\mathbf x), \ldots, f_n(\mathbf x)] \f$
     * the vector containing the original multiple objectives, with: \f$ \boldsymbol \lambda = (\lambda_1, \ldots, \lambda_n) \f$
     * a \f$n\f$-dimensional weight vector and with: \f$ \mathbf z^* = (z^*_1, \ldots, z^*_n) \f$
     * a-\f$n\f$ dimensional reference point. We also ussume \f$\lambda_i > 0, \forall i=1..n\f$ and \f$\sum_i \lambda_i = 1\f$
     *
     * The decomposed problem is thus a single objective optimization problem having the following single objective,
     * according to the decomposition method chosen:
     *
     * - weighted decomposition: \f$ f_d(\mathbf x) = \boldsymbol \lambda \cdot \mathbf f \f$
     *
     * - Tchebycheff decomposition: \f$ f_d(\mathbf x) = \max_{1 \leq i \leq m} \lambda_i \vert f_i(\mathbf x) - z^*_i \vert \f$
     *
     * - boundary interception method (with penalty constraint): \f$ f_d(\mathbf x) = d_1 + \theta d_2\f$
     *
     * where \f$d_1 = (\mathbf f - \mathbf z^*) \cdot \hat {\mathbf i}_{\lambda}\f$,
     * \f$d_2 = \vert (\mathbf f - \mathbf z^*) - d_1 \hat {\mathbf i}_{\lambda})\vert\f$ and
     * \f$ \hat {\mathbf i}_{\lambda} = \frac{\boldsymbol \lambda}{\vert \boldsymbol \lambda \vert}\f$
     *
     * @note The reference point \f$z^*\f$ is often taken as the ideal point and as such
     * it may be allowed to change during the course of the optimization / evolution. The argument adapt_ideal activates
     * this behaviour so that whenever a new ideal point is found \f$z^*\f$ is adapted accordingly.
     *
     * @note The use of pagmo::decompose discards gradients and hessians so that if the original user defined problem
     * implements them, they will not be available in the decomposed problem. The reason for this behaviour is that
     * the Tchebycheff decomposition is not differentiable. Also, the use of this class was originally intended for
     * derivative-free optimization.
     *
     * @see "Q. Zhang -- MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition"
     * @see https://en.wikipedia.org/wiki/Multi-objective_optimization#Scalarizing_multi-objective_optimization_problems
     */
class decompose : public problem
{
public:
    // default Constructor, only for serialization
    decompose () : problem(zdt(1u,2u)), m_weight({0.5, 0.5}), m_z({0.,0.}), m_method("weighted"), m_adapt_ideal(false)
    {}
    /// Constructor from problem
    /**
     * Wraps a user-defined problem so that its fitness will be decomposed using one of three decomposition methods.
     * pagmo::translate objects can be used as user-defined problems in the construction of a pagmo::problem.
     *
     * @tparam T Any type from which pagmo::problem is constructable
     * @param[in] p The input problem.
     * @param[in] method an std::string containing the decomposition method chosen
     * @param[in] weight the vector of weights \f$\boldsymbol \lambda\f$
     * @param[in] z the reference point \f$\mathbf z^*\f$
     * @param[in] adapt_ideal when true the reference point is adapted at each fitness evaluation to be the ideal point
     *
     * @throws std::invalid_argument if the problem of type \p T is single objective
     * @throws std::invalid_argument if the problem of type \p T is constrained
     * @throws std::invalid_argument if \p method is not one of ["weighted", "tchebycheff" or "bi"]
     * @throws std::invalid_argument if \p weight is not of size \f$n\f$
     * @throws std::invalid_argument if \p z is not of size \f$n\f$
     * @throws std::invalid_argument if \p weight is not such that \f$\lambda_i > 0, \forall i=1..n\f$
     * @throws std::invalid_argument if \p weight is not such that \f$\sum_i \lambda_i = 1\f$
     *
     */
    template <typename T>
    explicit decompose(T &&p, const vector_double &weight, const vector_double &z, const std::string &method = "weighted", bool adapt_ideal = false) :
            problem(std::forward<T>(p)), m_weight(weight), m_z(z), m_method(method), m_adapt_ideal(adapt_ideal)
    {
        auto original_fitness_dimension = static_cast<const problem*>(this)->get_nobj();
        // 0 - we check that the problem is multiobjective and unconstrained
        if (original_fitness_dimension < 2u) {
            pagmo_throw(std::invalid_argument, "Decomposition can only be applied to multi-objective problems, it seems you are trying to decompose a problem with " + std::to_string(original_fitness_dimension) + " objectives");
        }
        if (static_cast<const problem*>(this)->get_nc() != 0u) {
            pagmo_throw(std::invalid_argument, "Decomposition can only be applied to unconstrained problems, it seems you are trying to decompose a problem with " + std::to_string(static_cast<const problem*>(this)->get_nc()) + " constraints");
        }
        // 1 - we check that the decomposition method is one of "weighted", "tchebycheff" or "bi"
        if (method != "weighted" && method != "tchebycheff" && method != "bi") {
            pagmo_throw(std::invalid_argument, "Decomposition method requested is: " + method + " while only one of ['weighted', 'tchebycheff' or 'bi'] are allowed");
        }
        // 2 - we check the sizes of the input weight vector and of the reference point and forbids inf and nan
        if (weight.size() != original_fitness_dimension) {
            pagmo_throw(std::invalid_argument, "Weight vector size must be equal to the number of objectives. The size of the weight vector is " + std::to_string(weight.size()) + " while the problem has " + std::to_string(original_fitness_dimension) + " objectives");
        }
        for (auto item : weight)
        {
            if (!std::isfinite(item)) {
                pagmo_throw(std::invalid_argument, "Weight contains non finite numbers");
            }
        }
        if (z.size() != original_fitness_dimension) {
            pagmo_throw(std::invalid_argument, "Reference point size must be equal to the number of objectives. The size of the reference point is " + std::to_string(z.size()) + " while the problem has " + std::to_string(original_fitness_dimension) + " objectives");
        }
        for (auto item : z)
        {
            if (!std::isfinite(item)) {
                pagmo_throw(std::invalid_argument, "Reference point contains non finite numbers");
            }
        }

        // 3 - we check that the weight vector is normalized.
        auto sum = std::accumulate(weight.begin(), weight.end(), 0.);
        if (std::abs(sum-1.0) > 1E-8) {
            pagmo_throw(std::invalid_argument,"The weight vector must sum to 1 with a tolerance of 1E-8. The sum of the weight vector components was detected to be: " + std::to_string(sum));
        }
        // 4 - we check the weight vector only contains positive numbers
        for (decltype(m_weight.size()) i = 0u; i < m_weight.size(); ++i) {
            if (m_weight[i] < 0.) {
                pagmo_throw(std::invalid_argument,"The weight vector may contain only non negative values. A value of " + std::to_string(m_weight[i]) + " was detected at index " + std::to_string(i));
            }
        }
    }
    /// Fitness of the decomposed problem
    vector_double fitness(const vector_double &x) const
    {
        // we compute the fitness of the original multiobjective problem
        auto f = original_fitness(x);
        // if necessary we update the reference point
        if (m_adapt_ideal) {
            for (decltype(f.size()) i = 0u; i < f.size(); ++i) {
                if (f[i] < m_z[i])
                {
                    m_z[i] = f[i]; // its mutable so its ok
                }
            }
        }
        // we return the decomposed fitness
        return decompose_fitness(f, m_weight, m_z);
    }
    /// Fitness of the original problem
    /**
     * Returns the fitness of the original multi-objective problem used to construct the decomposed problem.
     *
     * @note This is *not* the fitness of the decomposed problem, that is returned by calling decompose::fitness()
     *
     * @param[in] x A decision vector
     *
     * @returns the fitness of the original multi-objective problem
     */
    vector_double original_fitness(const vector_double &x) const
    {
        // We call the fitness of the original multiobjective problem
        return static_cast<const problem*>(this)->fitness(x);
    }
    /// Decomposes a fitness vector
    /**
     * Returns the decomposed fitness vector.
     *
     * @param[in] f Input fitness
     * @param[in] weight the weight to be used in the decomposition
     * @param[in] ref_z the reference point to be used if either "tchebycheff" or "bi"
     * was indicated as a decomposition method
     * @returns the decomposed fitness vector
     * @throws std::invalid_argument if the input \p f, \p weight and \p ref_z have different sizes
     */
    vector_double decompose_fitness(const vector_double &f, const vector_double &weight, const vector_double &ref_z) const
    {
        if (weight.size() != f.size()) {
            pagmo_throw(std::invalid_argument, "Weight vector size must be equal to the number of objectives. The size of the weight vector is " + std::to_string(weight.size()) + " while " + std::to_string(f.size()) + " objectives were detected");
        }
        if (ref_z.size() != f.size()) {
            pagmo_throw(std::invalid_argument, "Reference point size must be equal to the number of objectives. The size of the reference point is " + std::to_string(ref_z.size()) + " while " + std::to_string(f.size()) + " objectives were detected");
        }
        double fd = 0.;
        if(m_method == "weighted") {
            for(decltype(f.size()) i = 0u; i < f.size(); ++i) {
                fd += weight[i] * f[i];
            }
        } else if (m_method == "tchebycheff") {
            double tmp,fixed_weight;
            for(decltype(f.size()) i = 0u; i < f.size(); ++i) {
               (weight[i] == 0.) ? (fixed_weight = 1e-4) : (fixed_weight = weight[i]); //fixes the numerical problem of 0 weights
               tmp = fixed_weight * std::abs(f[i] - ref_z[i]);
               if(tmp > fd) {
                   fd = tmp;
               }
           }
       } else if (m_method == "bi") { //BI method
            const double THETA = 5.;
            double d1 = 0.;
            double weight_norm = 0.;
            for(decltype(f.size()) i = 0u; i < f.size(); ++i) {
                d1 += (f[i] - ref_z[i]) * weight[i];
                weight_norm += std::pow(weight[i],2);
            }
            weight_norm = std::sqrt(weight_norm);
            d1 = d1 / weight_norm;

            double d2 = 0.;
            for(decltype(f.size()) i = 0u; i < f.size(); ++i) {
                d2 += std::pow(f[i] - (ref_z[i] + d1 * weight[i] / weight_norm), 2);
            }
            d2 = std::sqrt(d2);
            fd = d1 + THETA * d2;
        }
        return {fd};
    }
    /// A decomposed problem has one objective
    vector_double::size_type get_nobj() const
    {
        return 1u;
    }
    /// Returns a dense gradient_sparsity for the decomposed problem
    sparsity_pattern gradient_sparsity() const
    {
        return detail::dense_gradient(1u, static_cast<const problem*>(this)->get_nx());
    }
    /// Returns a dense hessians_sparsity for the decomposed problem
    std::vector<sparsity_pattern> hessians_sparsity() const
    {
        return detail::dense_hessians(1u, static_cast<const problem*>(this)->get_nx());
    }
    /// Gets the current reference point (may be different from the one the object was constructed with)
    vector_double get_z() const
    {
        return m_z;
    }
    /// Appends "[decomposed]" to the user-defined problem name
    std::string get_name() const
    {
        return static_cast<const problem*>(this)->get_name() + " [decomposed]";
    }
    /// Extra informations
    std::string get_extra_info() const
    {
        std::ostringstream oss;
        stream(oss,
            "\n\tDecomposition method: ", m_method,
            "\n\tDecomposition weight: ", m_weight,
            "\n\tDecomposition reference: ", m_z,
            "\n\tIdeal point adaptation: ", m_adapt_ideal
        );
        return static_cast<const problem*>(this)->get_extra_info() + oss.str();
    }

    /// Serialize
    template <typename Archive>
    void serialize(Archive &ar)
    {
        ar(cereal::base_class<problem>(this), m_weight, m_z, m_method, m_adapt_ideal);
    }

private:
    // Delete all that we do not want to inherit from problem
    // A - Common to all meta
    template <typename T>
    bool is() const = delete;
    vector_double::size_type get_nx() const = delete;
    vector_double::size_type get_nf() const = delete;
    vector_double::size_type get_nc() const = delete;
    unsigned long long get_fevals() const = delete;
    unsigned long long get_gevals() const = delete;
    unsigned long long get_hevals() const = delete;
    vector_double::size_type get_gs_dim() const = delete;
    std::vector<vector_double::size_type> get_hs_dim() const = delete;
    bool is_stochastic() const = delete;
    template <typename Archive>
    void save(Archive &) const = delete;
    template <typename Archive>
    void load(Archive &) = delete;

    // B - Specific to the decompose
    // A decomposed problem does not have gradients (Tchebycheff is not differentiable)
    vector_double gradient(const vector_double &dv) const = delete;
    // deleting has_gradient allows the automatic detection of gradients to see that decompose does not have any regardless
    // of whether the class its build from has them. A decompose problem will thus never have gradients
    bool has_gradient() const = delete;
    // deleting has_gradient_sparsity allows the automatic detection of gradient_sparsisty to see that decompose does have an
    // implementation for it. The sparsity will thus always be dense (decompose::gradient_sparsity) and referred to
    // a problem with one objective
    bool has_gradient_sparsity() const = delete;
    // A decomposed problem does not have hessians (Tchebycheff is not differentiable)
    std::vector<vector_double> hessians(const vector_double &dv) const = delete;
    // deleting has_hessians allows the automatic detection of hessians to see that decompose does not have any regardless
    // of whether the class its build from has them. A decompose problem will thus never have hessians
    bool has_hessians() const = delete;
    // deleting has_hessians_sparsity allows the automatic detection of hessians_sparsisty to see that decompose does have an
    // implementation for it. The hessians_sparsisty will thus always be dense (decompose::hessians_sparsisty) and referred to
    // a problem with one objective
    bool has_hessians_sparsity() const = delete;

    // decomposition weight
    vector_double m_weight;
    // decomposition reference point (only relevant/used for tchebycheff and boundary interception)
    mutable vector_double m_z;
    // decomposition method
    std::string m_method;
    // adapts the decomposition reference point whenever a better point is computed
    bool m_adapt_ideal;
};

}

PAGMO_REGISTER_PROBLEM(pagmo::decompose)

#endif
