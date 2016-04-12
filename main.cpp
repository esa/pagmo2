#include "include/algorithm.hpp"
#include "include/algorithms/null.hpp"
#include "include/problem.hpp"

using namespace pagmo;
struct example0_g
{
    // Mandatory, computes ... well ... the fitness
    vector_double fitness(const vector_double &x) const
    {
        return {x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + x[3]*x[3]};
    }

    // Optional, computes the gradient. In this simple case
    // df0/dx0, df0/dx1, df0/dx2, df0/dx3
    vector_double gradient(const vector_double &x) const
    {
        return {2 * x[0],2 * x[1], 2 * x[2], 2 * x[3]};
    }

    // Optional. Returns the sparsity of the problem as a sparsity_pattern
    // that is a std::vector<std::pair<long,long>> containing pairs
    // (i,j) indicating that the j-th variable "influences" the i-th component
    // in the fitness. When not implemented a dense problem is assumed
    sparsity_pattern gradient_sparsity() const
    {
        return {{0,0},{0,1},{0,2},{0,3}};
    }

    // Optional, computes the Hessians of the various fitness
    // components. That is d^2fk/dxidxj. In this case we have only
    // one fitness component, thus we only need one Hessian which is
    // also sparse as most of its components are 0.
    std::vector<vector_double> hessians(const vector_double &) const
    {
        return {{2.,2.,2.,2.}};
    }

    // Optional, computes the sparsity of the hessians.
    std::vector<sparsity_pattern> hessians_sparsity() const
    {
        return {{{0,0},{1,1},{2,2},{3,3}}};
    }

    // Mandatory, returns the dimension of the decision vector,
    // in this case fixed to 4
    vector_double::size_type get_n() const
    {
        return 4u;
    }

    // Mandatory, returns the dimension of the decision vector,
    // in this case fixed to 1 (single objective)
    vector_double::size_type get_nf() const
    {
        return 1u;
    }
    
    // Mandatory, returns the box-bounds
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{-10,-10,-10,-10},{10,10,10,10}};
    }

    // Optional, provides a name for the problem overrding the default name
    std::string get_name() const
    {   
        return std::string("My Problem");
    }

    // Optional, provides extra information that will be appended after
    // the default stream operator
    std::string extra_info() const {
        std::ostringstream s;
        s << "This is a simple toy problem with one fitness, " << '\n';
        s << "no constraint and a fixed dimension of 4." << "\n";
        s << "The fitness function gradient and hessians are also implemented" << "\n";
        s << "The sparsity of the gradient and hessians is user provided" << "\n";
        return s.str();
    }
    
    // Optional methods-data can also be accessed later via 
    // the problem::extract() method
    std::vector<vector_double> best_known() const
    {
        return {{0,0,0,0}};
    }
};

int main()
{
    algorithm a{algorithms::null{}};
    std::stringstream ss;
    {
    cereal::JSONOutputArchive oarchive(ss);
    oarchive(a);
    }
    std::cout << ss.str() << '\n';
    {
    cereal::JSONInputArchive iarchive(ss);
    iarchive(a);
    }    
    std::cout << a.extract<algorithms::null>()->get_a() << std::endl;
    a.evolve();

    problem p{example0_g{}};
    std::cout << p.extract<example0_g>()->fitness(std::vector<double>(4))[0] << std::endl;
}
