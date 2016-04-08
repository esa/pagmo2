#include <iostream>

#include "include/problem.hpp"

using namespace pagmo;

struct example0
{
    fitness_vector fitness(const decision_vector &x)
    {
        fitness_vector retval(1);
        retval[0] = x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + x[3]*x[3];
        return retval;
    }

    decision_vector::size_type get_n() const
    {
        return 4u;
    }

    fitness_vector::size_type get_nf() const
    {
        return 1u;
    }
    
    std::pair<decision_vector, decision_vector> get_bounds() const
    {
        decision_vector lb{1,1,1,1};
        decision_vector ub{5,5,5,5};
        return std::pair<decision_vector, decision_vector>(std::move(lb), std::move(ub));
    }

    std::string get_name() const
    {   
        return std::string("My Problem");
    }

    std::string extra_info() const {
        std::ostringstream s;
        s << "This is just a simple toy problem with one fitness, " << '\n';
        s << "no constraint and a fixed dimension of 4." << "\n";
        return s.str();
    }
    
    std::vector<decision_vector> best_known() const
    {
        return std::vector<decision_vector>{decision_vector{0,0,0,0}};
    }

};

// Problem with one objective one equality and one inequality constraint
struct example1
{
    fitness_vector fitness(const decision_vector &x)
    {
        fitness_vector retval(3);
        retval[0] = x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2];
        retval[1] = x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + x[3]*x[3] - 40;
        retval[2] = x[0] * x[1] * x[2] * x[3] + 25;
        return retval;
    }

    decision_vector::size_type get_n() const
    {
        return 4u;
    }

    fitness_vector::size_type get_nf() const
    {
        return 1u;
    }

    decision_vector::size_type get_nec() const
    {
        return 1u;
    }

    decision_vector::size_type get_nic() const
    {
        return 1u;
    }

    std::pair<decision_vector, decision_vector> get_bounds() const
    {
        decision_vector lb{1,1,1,1};
        decision_vector ub{5,5,5,5};
        return std::pair<decision_vector, decision_vector>(lb, ub);
    }
};

// Invalid problem (compile time error if used to construct a pagmo::problem).
struct example2 {};

int main()
{
    problem p0{example0{}};
    std::cout << p0 << '\n';
    std::cout << p0.get_nec() << '\n';
    std::cout << p0.get_nic() << '\n';
    pagmo::io::print(p0.get_bounds(), "\n");
    pagmo::io::print("fevals: ", p0.get_fevals(), "\n");
    pagmo::io::print(p0.fitness(decision_vector{2,2,2,2}), "\n");
    pagmo::io::print("fevals: ", p0.get_fevals(), "\n");
    problem p1{example1{}};
    std::cout << p1.get_nec() << '\n';
    std::cout << p1.get_nic() << '\n';
    // Compile time error if uncommented.
    // problem p2{example2{}};
    problem p2{std::move(p1)};
}
