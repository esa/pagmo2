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
};

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
};

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

};

