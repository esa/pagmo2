#ifndef PAGMO_PROBLEM_SHIFTED
#define PAGMO_PROBLEM_SHIFTED

#include <algorithm>
#include <cassert>

#include "../io.hpp"
#include "../problem.hpp"
#include "../types.hpp"
#include "../exceptions.hpp"
#include "null.hpp"

namespace pagmo
{


class shifted
{
public:
    shifted(const problem &p = problem{null{}}, const vector_double &shift = {1.}) : m_p(p), m_shift(shift)
    {
        if (shift.size() != p.get_nx()) {
            pagmo_throw(std::invalid_argument,"Length of shift vector is: " + std::to_string(shift.size()) + " while the problem dimension is: " + std::to_string(p.get_nx()));
        }
    }

    /// Fitness
    vector_double fitness(const vector_double &x) const
    {
        vector_double x_deshifted = deshift(x);
        return m_p.fitness(x_deshifted);
    }

    /// Number of objectives is unchanged
    vector_double::size_type get_nobj() const
    {
        return m_p.get_nobj();
    }

    /// Equality constraint dimension is unchanged
    vector_double::size_type get_nec() const
    {
        return m_p.get_nec();
    }

    /// Inequality constraint dimension is unchanged
    vector_double::size_type get_nic() const
    {
        return m_p.get_nic();
    }
    
    /// Problem bounds 
    std::pair<vector_double, vector_double> get_bounds() const
    {
        auto b_sh = m_p.get_bounds();
        return {shift(b_sh.first), shift(b_sh.second)};
    }

    /// Gradients
    vector_double gradient(const vector_double &x) const
    {
        vector_double x_deshifted = deshift(x);
        return m_p.gradient(x_deshifted);
    }

    /// Gradient sparsity
    sparsity_pattern gradient_sparsity() const
    {
        return m_p.gradient_sparsity();
    }

    /// Hessians
    std::vector<vector_double> hessians(const vector_double &x) const
    {
        vector_double x_deshifted = deshift(x);
        // TODO should we check if m_p has hessians?
        return m_p.hessians(x_deshifted);
    }

    /// Hessian sparsity
    std::vector<sparsity_pattern> hessians_sparsity() const
    {
        return m_p.hessians_sparsity();
    }

    /// Problem name
    std::string get_name() const
    {   
        return m_p.get_name() + " [shifted]";
    }

    /// Extra informations
    std::string get_extra_info() const
    {
        std::ostringstream oss;
        stream(oss, m_shift);
        return m_p.get_extra_info() + "\n\tShift Vector: " + oss.str();
    }
    
    /**
     * Returns the de-shifted version of the decision vector
     *
     * @param[in] dv decision vector to be de-shifted
     */
    vector_double deshift(const vector_double& x) const
    {
        assert(x.size() == m_shift.size());
        vector_double x_sh(x.size());
        std::transform(x.begin(), x.end(), m_shift.begin(), x_sh.begin(), std::minus<>());
        return x_sh;
    }

    vector_double shift(const vector_double& x) const
    {
        assert(x.size() == m_shift.size());
        vector_double x_sh(x.size());
        std::transform(x.begin(), x.end(), m_shift.begin(), x_sh.begin(), std::plus<>());
        return x_sh;
    }

    const vector_double& get_shift_vector() const 
    {
        return m_shift;
    }

    /// Serialization
    template <typename Archive>
    void serialize(Archive &ar) 
    {
        ar(m_p, m_shift);
    }

    // Ovverrides making sure the meta-problem is correctly identified
    // to have or not gradient etc. according to whether m_p has them.
    bool has_gradient() const
    {
        return m_p.has_gradient();
    }
    bool has_hessians() const
    {
        return m_p.has_hessians();
    }
    bool has_gradient_sparsity() const
    {
        return m_p.has_gradient_sparsity();
    }
    bool has_hessians_sparsity() const
    {
        return m_p.has_hessians_sparsity();
    }

private:
    problem m_p;
    vector_double m_shift;
};

}

PAGMO_REGISTER_PROBLEM(pagmo::shifted)

#endif
