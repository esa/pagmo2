// In this tutorial we implement the simple problem of minimizing
// f = x1^2 + x2^2 + x3^2 + x4^2 in the bounds:
// -1 <= xi <= 1

// All we need to do is to implement a struct (or class) having the
// following mandatory methods: 
//
// fitness_vector fitness(const decision_vector &)
// decision_vector::size_type get_n() const
// fitness_vector::size_type get_nf() const
// std::pair<decision_vector, decision_vector> get_bounds() const

#include "../include/types.hpp"

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
        decision_vector lb{-1,-1,-1,-1};
        decision_vector ub{ 1, 1, 1, 1};
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