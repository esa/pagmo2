#ifndef PAGMO_PROBLEM_SHIFTED
#define PAGMO_PROBLEM_SHIFTED

#include <algorithm>
#include <cassert>

#include "../problem.hpp"
#include "../types.hpp"
#include "../exceptions.hpp"
#include "null.hpp"

namespace pagmo
{

/// The translate meta-problem
/**
 * This pagmo::problem translates the whole search space of an input 
 * pagmo::problem by a fixed translation vector. 
 */
class translate
{
public:
    /// Default constructor
    translate():m_p(problem{null{}}),m_translation({1.}) {}

    /// Constructor from user-defined problem and translation vector
    /**
     * Constructs a pagmo::problem translating a pagmo::problem 
     * by a translation vector.
     *
     * @tparam T Any type from which pagmo::problem is constructable
     * @param[in] p The user defined problem.
     * @param[in] translation An std::vector containing the translation to apply.
     *
     * @throws std::invalid_argument if the length of \p translation is 
     * not equal to the problem dimension \f$ n_x\f$.
     * @throws unspecified any exception thrown by the pagmo::problem constructor
     */
    template <typename T>
    explicit translate(T &&p, const vector_double &translation) : m_p(std::forward<T>(p)), m_translation(translation)    {
        if (translation.size() != m_p.get_nx()) {
            pagmo_throw(std::invalid_argument,"Length of shift vector is: " + std::to_string(translation.size()) + " while the problem dimension is: " + std::to_string(m_p.get_nx()));
        }
    }

    /// Fitness of the translated problem
    vector_double fitness(const vector_double &x) const
    {
        vector_double x_deshifted = translate_back(x);
        return m_p.fitness(x_deshifted);
    }

    /// Number of objectives (unchanged)
    vector_double::size_type get_nobj() const
    {
        return m_p.get_nobj();
    }

    /// Equality constraint dimension (unchanged)
    vector_double::size_type get_nec() const
    {
        return m_p.get_nec();
    }

    /// Inequality constraint dimension (unchanged)
    vector_double::size_type get_nic() const
    {
        return m_p.get_nic();
    }
    
    /// Problem bounds of the translated problem
    std::pair<vector_double, vector_double> get_bounds() const
    {
        auto b_sh = m_p.get_bounds();
        return {apply_translation(b_sh.first), apply_translation(b_sh.second)};
    }

    /// Gradients of the translated problem
    vector_double gradient(const vector_double &x) const
    {
        vector_double x_deshifted = translate_back(x);
        return m_p.gradient(x_deshifted);
    }

    /// Gradient sparsity of the translated problem
    sparsity_pattern gradient_sparsity() const
    {
        return m_p.gradient_sparsity();
    }

    /// Hessians of the translated problem
    std::vector<vector_doubl e> hessians(const vector_double &x) const
    {
        vector_double x_deshifted = translate_back(x);
        return m_p.hessians(x_deshifted);
    }

    /// Hessian sparsity of the translated problem
    std::vector<sparsity_pattern> hessians_sparsity() const
    {
        return m_p.hessians_sparsity();
    }

    /// Appends "[shifted]" to the user-defined problem name
    std::string get_name() const
    {   
        return m_p.get_name() + " [shifted]";
    }

    /// Extra informations
    std::string get_extra_info() const
    {
        std::ostringstream oss;
        stream(oss, m_translation);
        return m_p.get_extra_info() + "\n\tTranslation Vector: " + oss.str();
    }
    
    /// Gets the translation vector
    const vector_double& get_translation() const 
    {
        return m_translation;
    }

    /// Serialization
    template <typename Archive>
    void serialize(Archive &ar) 
    {
        ar(m_p, m_translation);
    }

    /// Sets problem::has_gradient to the value of the user-defined problem's
    bool has_gradient() const
    {
        return m_p.has_gradient();
    }

    /// Sets problem::has_hessians to the value of the user-defined problem's
    bool has_hessians() const
    {
        return m_p.has_hessians();
    }

    /// Sets problem::has_gradient_sparsity to the value of the user-defined problem's
    bool has_gradient_sparsity() const
    {
        return m_p.has_gradient_sparsity();
    }

    /// Sets problem::has_hessians_sparsity to the value of the user-defined problem's
    bool has_hessians_sparsity() const
    {
        return m_p.has_hessians_sparsity();
    }

private:
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

    problem m_p;
    vector_double m_translation;
};

}

PAGMO_REGISTER_PROBLEM(pagmo::translate)

#endif
