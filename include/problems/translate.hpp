#ifndef PAGMO_PROBLEM_TRANSLATE_HPP
#define PAGMO_PROBLEM_TRANSLATE_HPP

#include <algorithm>
#include <cassert>

#include "../exceptions.hpp"
#include "../problem.hpp"
#include "../serialization.hpp"
#include "../types.hpp"
#include "null_problem.hpp"

namespace pagmo
{

/// The translate meta-problem
/**
 * This meta-problem translates the whole search space of an input problem
 * by a fixed translation vector
 */
class translate : public problem
{
public:
    /// Default constructor
    translate() : problem(null_problem{}), m_translation({1.}) {}

    /// Constructor from user-defined problem and translation vector
    /**
     * Wraps a user-defined problem so that its fitness , bounds, .... etc. will be shifted by a translation vector.
     * pagmo::translate objects can be used as user-defined problems in the construction of a pagmo::problem.
     *
     * @tparam T Any type from which pagmo::problem is constructable
     * @param[in] p The user defined problem.
     * @param[in] translation An <tt>std::vector</tt> containing the translation to apply.
     *
     * @throws std::invalid_argument if the length of \p translation is
     * not equal to the problem dimension \f$ n_x\f$.
     * @throws unspecified any exception thrown by the pagmo::problem constructor
     */
    template <typename T>
    explicit translate(T &&p, const vector_double &translation) : problem(std::forward<T>(p)), m_translation(translation) {
        if (translation.size() != static_cast<const problem*>(this)->get_nx()) {
            pagmo_throw(std::invalid_argument,"Length of shift vector is: " + std::to_string(translation.size()) + " while the problem dimension is: " + std::to_string(static_cast<const problem*>(this)->get_nx()));
        }
    }

    /// Fitness of the translated problem
    vector_double fitness(const vector_double &x) const
    {
        vector_double x_deshifted = translate_back(x);
        return static_cast<const problem*>(this)->fitness(x_deshifted);
    }

    /// Problem bounds of the translated problem
    std::pair<vector_double, vector_double> get_bounds() const
    {
        auto b_sh = static_cast<const problem*>(this)->get_bounds();
        return {apply_translation(b_sh.first), apply_translation(b_sh.second)};
    }

    /// Gradients of the translated problem
    vector_double gradient(const vector_double &x) const
    {
        vector_double x_deshifted = translate_back(x);
        return static_cast<const problem*>(this)->gradient(x_deshifted);
    }


    /// Hessians of the translated problem
    std::vector<vector_double> hessians(const vector_double &x) const
    {
        vector_double x_deshifted = translate_back(x);
        return static_cast<const problem*>(this)->hessians(x_deshifted);
    }

    /// Appends "[translated]" to the user-defined problem name
    std::string get_name() const
    {
        return static_cast<const problem*>(this)->get_name() + " [translated]";
    }

    /// Extra informations
    std::string get_extra_info() const
    {
        std::ostringstream oss;
        stream(oss, m_translation);
        return static_cast<const problem*>(this)->get_extra_info() + "\n\tTranslation Vector: " + oss.str();
    }

    /// Gets the translation vector
    const vector_double& get_translation() const
    {
        return m_translation;
    }

    /// Serialize
    template <typename Archive>
    void serialize(Archive &ar)
    {
        ar(cereal::base_class<problem>(this), m_translation);
    }

private:
    // Delete all that we do not want to inherit from problem
    // A - Common to all meta
    template <typename T>
    const T *extract() const = delete;
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

    vector_double translate_back(const vector_double& x) const
    {
        assert(x.size() == m_translation.size());
        vector_double x_sh(x.size());
        std::transform(x.begin(), x.end(), m_translation.begin(), x_sh.begin(), std::minus<>());
        return x_sh;
    }

    vector_double apply_translation(const vector_double& x) const
    {
        assert(x.size() == m_translation.size());
        vector_double x_sh(x.size());
        std::transform(x.begin(), x.end(), m_translation.begin(), x_sh.begin(), std::plus<>());
        return x_sh;
    }
    /// translation vector
    vector_double m_translation;
};

}

PAGMO_REGISTER_PROBLEM(pagmo::translate)

#endif
