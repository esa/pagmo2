#include "../include/problem.hpp"

#define BOOST_TEST_MODULE pagmo_problem_test
#include <boost/test/unit_test.hpp>

#include <stdexcept>
#include <utility>

#include "../include/types.hpp"

using namespace pagmo;

struct base_p
{

    base_p(
        unsigned int dim = 1, 
        unsigned int nobj = 1, 
        unsigned int nec = 0, 
        unsigned int nic = 0, 
        const vector_double &lb = vector_double(1,0.), 
        const vector_double &ub = vector_double(1,1.)
    ) : m_dim(dim), m_nobj(nobj), m_nec(nec), m_nic(nic), 
        m_lb(lb), m_ub(ub) {}

    vector_double fitness(const vector_double &) const
    {
        return vector_double(m_nobj,0.);
    }
    vector_double::size_type get_n() const
    {
        return m_dim;
    }
    vector_double::size_type get_nobj() const
    {
        return m_nobj;
    }
    vector_double::size_type get_nec() const
    {
        return m_nec;
    }
    vector_double::size_type get_nic() const
    {
        return m_nic;
    }
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {m_lb,m_ub};
    }
    unsigned int m_dim;
    unsigned int m_nobj;
    unsigned int m_nec;
    unsigned int m_nic;
    vector_double m_lb;
    vector_double m_ub;
};

struct grad_p : base_p
{
    grad_p(
        unsigned int dim = 1, 
        unsigned int nobj = 1, 
        unsigned int nec = 0, 
        unsigned int nic = 0, 
        const vector_double &lb = vector_double(1,0.), 
        const vector_double &ub = vector_double(1,1.),
        const vector_double &g = vector_double(1,1.),
        const sparsity_pattern &gs = sparsity_pattern({{0,0}})
     ) : base_p(dim,nobj,nec,nic,lb,ub), m_g(g), m_gs(gs) {}

    vector_double gradient(const vector_double &) const
    {
        return m_g;
    }

    sparsity_pattern gradient_sparsity() const
    {
        return m_gs;
    }

    vector_double m_g;
    sparsity_pattern m_gs;
};

BOOST_AUTO_TEST_CASE(problem_construction_test)
{
    // We check that malformed problems throw std::invalid argument
    // 1 - lb > ub
    BOOST_CHECK_THROW(problem{base_p(2,1,0,0,{0.,0.},{1.,-1})}, std::invalid_argument);
    // 2 - lb length is wrong
    BOOST_CHECK_THROW(problem{base_p(2,1,0,0,{0.},{1., 2.})}, std::invalid_argument);
    // 3 - ub length is wrong
    BOOST_CHECK_THROW(problem{base_p(2,1,0,0,{0.,0.},{1., 2., 3.})}, std::invalid_argument);
    // 4 - gradient sparsity has index out of bounds
    BOOST_CHECK_THROW(problem{grad_p(2,1,0,0,{0.,0.},{1., 1.},{0,1},{{0,0},{3,4}})}, std::invalid_argument);
    // 5 - gradient sparsity has a repeating pair 
    BOOST_CHECK_THROW(problem{grad_p(2,1,0,0,{0.,0.},{1., 1.},{0,1},{{0,0},{0,0}})}, std::invalid_argument);
}
