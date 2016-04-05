using decision_vector = std::vector<double>;
using decision_vector_int = std::vector<long long>;
using fitness_vector = std::vector<double>;
using gradient_vector = std::vector<double>;
using sparsity_pattern std::vector<std::pair<long, long> >;

// Problem with one objective no constraints
struct example0
{
    fitness_vector fitness(const decision_vector &)
    {
        fitness_vector retval(1);
        retval[0] = x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2];
        return retval;
    }

    decision_vector::size_type get_n()
    {
        return 4u;
    }

    decision_vector::size_type get_nf()
    {
        return 1u;
    }
    
    std::pair<decision_vector, decision_vector> get_bounds()
    {
        decision_vector lb{1,1,1,1};
        decision_vector ub{5,5,5,5};
        return std::pair<decision_vector, decision_vector>(lb, ub);
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

    decision_vector::size_type get_n()
    {
        return 4u;
    }

    decision_vector::size_type get_nf()
    {
        return 1u;
    }

    decision_vector::size_type get_nec()
    {
        return 1u;
    }

    decision_vector::size_type get_nic()
    {
        return 1u;
    }

    std::pair<decision_vector, decision_vector> get_bounds()
    {
        decision_vector lb{1,1,1,1};
        decision_vector ub{5,5,5,5};
        return std::pair<decision_vector, decision_vector>(lb, ub);
    }
};

// Problem with one objective and gradients (dense)
struct example2
{
    fitness_vector fitness(const decision_vector &x)
    {
        fitness_vector retval(1);
        retval[0] = x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2];
        return retval;
    }

    std::vector<gradient_vector> fitness_gradient(const decision_vector &x)
    {
        gradient_vector retval(4);
        retval[0] = x[0] * x[3] + x[3] * (x[0] + x[1] + x[2]);
        retval[1] = x[0] * x[3];
        retval[2] = x[0] * x[3] + 1;
        retval[3] = x[0] * (x[0] + x[1] + x[2]);
        return std::vector<gradient_vector>(1, retval);
    }

    sparsity_pattern gradient_sparsity()  
    {
        sparsity_pattern retval;
        for (auto i, i<get_n(); ++i) {
            sparsity_pattern.push_back(std::pair<long, long>(0, i))
        }
        return retval;
    }

    decision_vector::size_type get_n()
    {
        return 4u;
    }

    decision_vector::size_type get_nf()
    {
        return 1u;
    }

    std::pair<decision_vector, decision_vector> get_bounds()
    {
        decision_vector lb{1,1,1,1};
        decision_vector ub{5,5,5,5};
        return std::pair<decision_vector, decision_vector>(lb, ub);
    }
};

// Problem with one objective one equality and one inequality constraint
// and gradients (dense). No Hessian.
struct example3
{
    fitness_vector fitness(const decision_vector &x)
    {
        fitness_vector retval(3);
        retval[0] = x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2];
        retval[1] = x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + x[3]*x[3] - 40;
        retval[2] = x[0] * x[1] * x[2] * x[3] + 25;
        return retval;
    }

    std::vector<gradient_vector> fitness_gradient(const decision_vector &x)
    {
        std::vector<gradient_vector> retval;

        // Objective
        gradient_vector objective_g(4);
        objective_g[0] = x[0] * x[3] + x[3] * (x[0] + x[1] + x[2]);
        objective_g[1] = x[0] * x[3];
        objective_g[2] = x[0] * x[3] + 1;
        objective_g[3] = x[0] * (x[0] + x[1] + x[2]);
        retval.push_back(objective_g);

        // Equality constraint
        gradient_vector constraint_e(4);
        constraint_e[0] = 2 * x[0];
        constraint_e[1] = 2 * x[1];
        constraint_e[2] = 2 * x[2];
        constraint_e[3] = 2 * x[3];
        retval.push_back(constraint_e);

        // Inequality constraint
        gradient_vector constraint_i(4);
        constraint_i[0] = x[1] * x[2] * x[3];
        constraint_i[1] = x[0] * x[2] * x[3];
        constraint_i[2] = x[1] * x[0] * x[3];
        constraint_i[3] = x[1] * x[2] * x[0];
        retval.push_back(constraint_i);

        return retval;
    }

    sparsity_pattern gradient_sparsity()  
    {
        sparsity_pattern retval;
        for (auto i, i<get_n(); ++i) {
            sparsity_pattern.push_back(std::pair<long, long>(0, i))
        }
        return retval;
    }

    decision_vector::size_type get_n()
    {
        return 4u;
    }

    decision_vector::size_type get_nf()
    {
        return 1u;
    }

    decision_vector::size_type get_nec()
    {
        return 1u;
    }

    decision_vector::size_type get_nic()
    {
        return 1u;
    }

    std::pair<decision_vector, decision_vector> get_bounds()
    {
        decision_vector lb{1,1,1,1};
        decision_vector ub{5,5,5,5};
        return std::pair<decision_vector, decision_vector>(lb, ub);
    }
};