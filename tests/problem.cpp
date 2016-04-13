#include "../include/problem.hpp"

#define BOOST_TEST_MODULE pagmo_problem_test
#include <boost/test/unit_test.hpp>

#include <stdexcept>
#include <utility>
#include <vector>

#include "../include/types.hpp"

using namespace pagmo;

// Generates a dummy simple problem with arbitrary dimensions and return values
struct base_p
{

    base_p(
        unsigned int dim = 1, 
        unsigned int nobj = 1, 
        unsigned int nec = 0, 
        unsigned int nic = 0, 
        const vector_double &ret_fit = {1},         
        const vector_double &lb = {0}, 
        const vector_double &ub = {1}
    ) : m_dim(dim), m_nobj(nobj), m_nec(nec), m_nic(nic), 
        m_lb(lb), m_ub(ub), m_ret_fit(ret_fit) {}

    vector_double fitness(const vector_double &) const
    {
        return m_ret_fit;
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
    vector_double m_ret_fit;
};

// Generates a dummy problem with arbitrary dimensions and return values
// having the gradient implemented
struct grad_p : base_p
{
    grad_p(
        unsigned int dim = 1, 
        unsigned int nobj = 1, 
        unsigned int nec = 0, 
        unsigned int nic = 0, 
        const vector_double &ret_fit = {1}, 
        const vector_double &lb = {0}, 
        const vector_double &ub = {1},
        const vector_double &g = {1},
        const sparsity_pattern &gs = {{0,0}}
     ) : base_p(dim,nobj,nec,nic,ret_fit,lb,ub), m_g(g), m_gs(gs) {}

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

// Generates a dummy problem with arbitrary dimensions and return values
// having the hessians implemented
struct hess_p : base_p
{
    hess_p(
        unsigned int dim = 1, 
        unsigned int nobj = 1, 
        unsigned int nec = 0, 
        unsigned int nic = 0, 
        const vector_double &ret_fit = {1}, 
        const vector_double &lb = {0}, 
        const vector_double &ub = {1},
        const std::vector<vector_double> &h = {{1}},
        const std::vector<sparsity_pattern> &hs = {{{0,0}}}
     ) : base_p(dim,nobj,nec,nic,ret_fit,lb,ub), m_h(h), m_hs(hs) {}

    std::vector<vector_double> hessians(const vector_double &) const
    {
        return m_h;
    }

    std::vector<sparsity_pattern> hessians_sparsity() const
    {
        return m_hs;
    }

    std::vector<vector_double> m_h;
    std::vector<sparsity_pattern> m_hs;
};

BOOST_AUTO_TEST_CASE(problem_construction_test)

{
    // We check that problems with inconsistent dimensions throw
    // std::invalid argument
    vector_double lb_2(2,0);
    vector_double ub_2(2,1);
    vector_double lb_3(3,1);
    vector_double ub_3(3,1);
    vector_double fit_1(1,1);
    vector_double fit_2(2,1);
    vector_double fit_12(12,11);
    vector_double lb_11(11,0);
    vector_double ub_11(11,0);

    vector_double grad_2{1,1};
    sparsity_pattern grads_2_outofbounds{{0,0},{3,4}};
    sparsity_pattern grads_2_repeats{{0,0},{0,0}};
    sparsity_pattern grads_2_correct{{0,0},{0,1}};

    std::vector<vector_double> hess_22{{1,1},{1,1}};
    std::vector<sparsity_pattern> hesss_22_outofbounds{{{0,0},{12,13}},{{0,0},{1,0}}};
    std::vector<sparsity_pattern> hesss_22_notlowertriangular{{{0,0},{0,1}},{{0,0},{1,0}}};
    std::vector<sparsity_pattern> hesss_22_repeated{{{0,0},{0,0}},{{0,0},{1,0}}};
    std::vector<sparsity_pattern> hesss_22_correct{{{0,0},{1,0}},{{0,0},{1,0}}};

    // 1 - lb > ub
    BOOST_CHECK_THROW(problem{base_p(2,1,0,0,fit_2,ub_2,lb_2)}, std::invalid_argument);
    // 2 - lb length is wrong
    BOOST_CHECK_THROW(problem{base_p(2,1,0,0,fit_2,lb_3,ub_2)}, std::invalid_argument);
    // 3 - ub length is wrong
    BOOST_CHECK_THROW(problem{base_p(2,1,0,0,fit_2,lb_2,ub_3)}, std::invalid_argument);
    // 4 - gradient sparsity has index out of bounds
    BOOST_CHECK_THROW(problem{grad_p(2,1,0,0,fit_2,lb_2,ub_2,grad_2, grads_2_outofbounds)}, std::invalid_argument);
    // 5 - gradient sparsity has a repeating pair 
    BOOST_CHECK_THROW(problem{grad_p(2,1,0,0,fit_2,lb_2,ub_2,grad_2, grads_2_repeats)}, std::invalid_argument);
    // 6 - hessian sparsity has index out of bounds
    BOOST_CHECK_THROW(problem{hess_p(2,1,1,0,fit_2,lb_2,ub_2,hess_22, hesss_22_outofbounds)}, std::invalid_argument);
    // 7 - hessian sparsity is not lower triangular
    BOOST_CHECK_THROW(problem{hess_p(2,1,1,0,fit_2,lb_2,ub_2,hess_22, hesss_22_notlowertriangular)}, std::invalid_argument);
    // 8 - hessian sparsity has repeated indexes
    BOOST_CHECK_THROW(problem{hess_p(2,1,1,0,fit_2,lb_2,ub_2,hess_22, hesss_22_repeated)}, std::invalid_argument);

    // We check that the data members are initialized correctly (i.e. counters to zero
    // and gradient / hessian dimensions to the right values
    {
        problem p1{base_p(2,2,0,0,fit_2,lb_2,ub_2)};
        problem p2{base_p(11,3,4,5,fit_12,lb_11,ub_11)};
        problem p3{grad_p(2,1,0,0,fit_2,lb_2,ub_2,grad_2, grads_2_correct)};
        problem p4{hess_p(2,1,1,0,fit_2,lb_2,ub_2,hess_22, hesss_22_correct)};
        BOOST_CHECK(p1.get_fevals() == 0u);
        BOOST_CHECK(p1.get_gevals() == 0u);
        BOOST_CHECK(p1.get_hevals() == 0u);
        BOOST_CHECK(p1.get_hevals() == 0u);
        // dense sparsity defined by default
        BOOST_CHECK(p1.get_gs_dim() == 4u);
        BOOST_CHECK((p1.get_hs_dim() == std::vector<vector_double::size_type>{3u, 3u}));
        BOOST_CHECK(p2.get_gs_dim() == 12u*11u);
        BOOST_CHECK((p2.get_hs_dim() == std::vector<vector_double::size_type>{66u, 66u, 66u, 66u, 66u, 66u, 66u, 66u, 66u, 66u, 66u, 66u}));
        // user defined sparsity
        BOOST_CHECK(p3.get_gs_dim() == 2u);
        BOOST_CHECK((p4.get_hs_dim() == std::vector<vector_double::size_type>{2u, 2u}));
    }
}


