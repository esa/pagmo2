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
class translate : problem
{
public:
    /// Default constructor
    translate() : problem(null{}), m_translation({1.}) {}

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
    //template <typename T>
    //explicit translate(T &&p, const vector_double &translation) : problem(p), m_translation(translation)    {
    //    if (translation.size() != get_nx()) {
    //        pagmo_throw(std::invalid_argument,"Length of shift vector is: " + std::to_string(translation.size()) + " while the problem dimension is: " + std::to_string(get_nx()));
    //    }
   // }

    /// Fitness of the translated problem
    vector_double fitness(const vector_double &x) const
    {
        vector_double x_deshifted = translate_back(x);
        return dynamic_cast<const problem*>(this)->fitness(x_deshifted);
    }

    ///
    vector_double::size_type get_nobj() const
    {
        return dynamic_cast<const problem*>(this)->get_nobj();
    }
    
    vector_double::size_type get_nec() const
    {
        return dynamic_cast<const problem*>(this)->get_nec();
    }

    vector_double::size_type get_nic() const
    {
        return dynamic_cast<const problem*>(this)->get_nic();
    }

    /// Problem bounds of the translated problem
    std::pair<vector_double, vector_double> get_bounds() const
    {
        auto b_sh = dynamic_cast<const problem*>(this)->get_bounds();
        return {apply_translation(b_sh.first), apply_translation(b_sh.second)};
    }

    /// Gradients of the translated problem
    vector_double gradient(const vector_double &x) const
    {
        vector_double x_deshifted = translate_back(x);
        return dynamic_cast<const problem*>(this)->gradient(x_deshifted);
    }


    /// Hessians of the translated problem
    std::vector<vector_double> hessians(const vector_double &x) const
    {
        vector_double x_deshifted = translate_back(x);
        return dynamic_cast<const problem*>(this)->hessians(x_deshifted);
    }


    /// Appends "[shifted]" to the user-defined problem name
    std::string get_name() const
    {   
        return dynamic_cast<const problem*>(this)->get_name() + " [shifted]";
    }

    /// Extra informations
    std::string get_extra_info() const
    {
        std::ostringstream oss;
        stream(oss, m_translation);
        return dynamic_cast<const problem*>(this)->get_extra_info() + "\n\tTranslation Vector: " + oss.str();
    }
    
    /// Gets the translation vector
    const vector_double& get_translation() const 
    {
        return m_translation;
    }

    /// Serialization
    //template <typename Archive>
    //void serialize(Archive &ar) 
    //{
    //    ar(m_translation);
    //}

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

    /// translation vector
    vector_double m_translation;
};

}

//PAGMO_REGISTER_PROBLEM(pagmo::translate)

#endif
