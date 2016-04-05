using decision_vector = std::vector<double>;
using decision_vector_int = std::vector<long long>;
using constraint_vector = std::vector<double>;
using fitness_vector = std::vector<double>;
using gradient_vector = std::vector<double>;
using index_vector = std::vector<long long>;
using sparsity_pattern std::pair<index_vector, index_vector>;


struct example0
{
    fitness_vector objfun(const decision_vector &)
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
    fitness_vector objfun(const decision_vector &x)
    {
        fitness_vector retval(1);
        retval[0] = x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2];
        return retval;
    }

    constraint_vector constraints(const decision_vector &x)
    {
        constraint_vector retval(2);
        retval[0] = x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + x[3]*x[3] - 40;
        retval[1] = x[0] * x[1] * x[2] * x[3] + 25;
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

    decision_vector::size_type get_nc()
    {
        return 2u;
    }
};

struct example2
{
    example2() : i_var(), j_fun()
    {
        compute_sparsity();
    }

    fitness_vector objfun(const decision_vector &x)
    {
        fitness_vector retval(1);
        retval[0] = x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2];
        return retval;
    }

    gradient_vector objfun_gradient(const decision_vector &x)
    {
        gradient_vector retval(4);
        retval[0] = x[0] * x[3] + x[3] * (x[0] + x[1] + x[2]);
        retval[1] = x[0] * x[3];
        retval[2] = x[0] * x[3] + 1;
        retval[3] = x[0] * (x[0] + x[1] + x[2]);
    }

    sparsity_pattern compute_sparsity()  
    {
        for (auto i, i<get_n(); ++i) {
            i_var.push_back(i);
            j_fun.push_back(0);
        }
    }

    decision_vector::size_type get_n()
    {
        return 4u;
    }

    decision_vector::size_type get_nf()
    {
        return 1u;
    }

    index_vector i_var;
    index_vector j_fun;
};