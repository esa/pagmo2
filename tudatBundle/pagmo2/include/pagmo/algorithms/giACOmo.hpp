/* Copyright 2017 PaGMO development team

This file is part of the PaGMO library.

The PaGMO library is free software; you can redistribute it and/or modify
it under the terms of either:

  * the GNU Lesser General Public License as published by the Free
    Software Foundation; either version 3 of the License, or (at your
    option) any later version.

or

  * the GNU General Public License as published by the Free Software
    Foundation; either version 3 of the License, or (at your option) any
    later version.

or both in parallel, as here.

The PaGMO library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.

You should have received copies of the GNU General Public License and the
GNU Lesser General Public License along with the PaGMO library.  If not,
see https://www.gnu.org/licenses/. */

#ifndef PAGMO_ALGORITHMS_GIACOMO_HPP
#define PAGMO_ALGORITHMS_GIACOMO_HPP

#include <algorithm> // std::shuffle, std::transform
#include <iomanip>
#include <numeric> // std::iota, std::inner_product
#include <random>
#include <string>
#include <tuple>

#include <pagmo/algorithm.hpp> // needed for the cereal macro
#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/problems/decompose.hpp>
#include <pagmo/rng.hpp>
#include <pagmo/utils/multi_objective.hpp> // crowding_distance, etc..
#include <pagmo/utils/generic.hpp> // uniform_real_from_range

namespace pagmo
{
/// Extended ACO
/**
 * \image html ACO.jpg "The ACO flowchart" width=3cm [TO BE ADDED]

 * ACO is inspired by the natural mechanism with which real ant colonies forage food.
 * This algorithm has shown promising results in many trajectory optimization problems.
 * The first appearance of the algorithm happened in Dr. Marco Dorigo's thesis, in 1992.
 * ACO generates future generations of ants by using the a multi-kernel gaussian distribution
 * based on three parameters (i.e., pheromone values) which are computed depending on the quality
 * of each previous solution. The solutions are ranked through an oracle penalty method.
 *
 *
 * The version implemented in pagmo can be applied to box-bounded multiple-objective optimization.
 *
 * See:  M. Schlueter, et al. (2009). Extended ant colony optimization for non-convex
 * mixed integer non-linear programming. Computers & Operations Research.
 */
class gi_aco_mo
{
public:
    /// Single entry of the log (gen, fevals, ideal_point)
    typedef std::tuple<unsigned, unsigned long long, vector_double> log_line_type;
    /// The log
    typedef std::vector<log_line_type> log_type;

    /// Constructor
    /**
    * Constructs the ACO user defined algorithm.
    *
    * @param[in] gen Generations: number of generations to evolve.
    * @param[in] acc Accuracy parameter: for inequality and equality constraints .
    * @param[in] fstop Objective stopping criterion: when the objective value reaches this value, the algorithm is stopped [for multi-objective, this applies to the first obj. only].
    * @param[in] impstop Improvement stopping criterion: if a positive integer is assigned here, the algorithm will count the runs without improvements, if this number will exceed IMPSTOP value, the algorithm will be stopped.
    * @param[in] evalstop Evaluation stopping criterion: same as previous one, but with function evaluations
    * @param[in] focus Focus parameter: this parameter makes the search for the optimum greedier and more focused on local improvements (the higher the greedier). If the value is very high, the search is more focused around the current best solutions
    * @param[in] ker Kernel: number of solutions stored in the solution archive
    * @param[in] oracle Oracle parameter: this is the oracle parameter used in the penalty method
    * @param[in] paretomax Max number of non-dominated solutions: this regulates the max number of Pareto points to be stored
    * @param[in] epsilon Pareto precision: the smaller this parameter, the higher the chances to introduce a new solution in the Pareto front
    * @param seed seed used by the internal random number generator (default is random)
    * @throws std::invalid_argument if \p acc is not \f$ \in [0,1[\f$, \p fstop is not positive, \p impstop is not a
    * positive integer, \p evalstop is not a positive integer, \p focus is not \f$ \in [0,1[\f$, \p ants is not a positive integer,
    * \p ker is not a positive integer, \p oracle is not positive, \p paretomax is not a positive integer, \p epsilon is not \f$ \in [0,1[\f$
    */
    gi_aco_mo(unsigned gen = 1u, double acc = 0.95, unsigned  fstop = 1, unsigned impstop = 1, unsigned evalstop = 1,
          double focus = 0.9,  unsigned ker = 10, double oracle=1.0, unsigned paretomax = 10,
            double epsilon = 0.9, unsigned seed = pagmo::random_device::next())
        : m_gen(gen), m_acc(acc), m_fstop(fstop), m_impstop(impstop), m_evalstop(evalstop), m_focus(focus),
          m_ker(ker), m_oracle(oracle), m_paretomax(paretomax), m_epsilon(epsilon), m_e(seed), m_seed(seed), m_verbosity(0u),
          m_log()
    {
        if (acc >= 1. || acc < 0.) {
            pagmo_throw(std::invalid_argument, "The accuracy parameter must be in the [0,1[ range, while a value of "
                                                   + std::to_string(acc) + " was detected");
        }
        if (focus >= 1. || focus < 0.) {
            pagmo_throw(std::invalid_argument,
                        "The focus parameter must be in the [0,1[ range, while a value of "
                            + std::to_string(focus) + " was detected");
        }
        if (oracle < 0.) {
            pagmo_throw(std::invalid_argument,
                        "The oracle parameter must be >=0, while a value of "
                            + std::to_string(oracle) + " was detected");
        }

        if (epsilon >= 1. || epsilon < 0.) {
            pagmo_throw(std::invalid_argument,
                        "The Pareto precision parameter must be in [0, 1[, while a value of "
                            + std::to_string(epsilon) + " was detected");
        }


    }

    /// Algorithm evolve method (juice implementation of the algorithm)
    /**
     *
     * Evolves the population for the requested number of generations.
     *
     * @param pop population to be evolved
     * @return evolved population
     * @throw std::invalid_argument if pop.get_problem() is stochastic, single objective or has non linear constraints.
     * If \p int_dim is larger than the problem dimension. If the population size is smaller than 5 or not a multiple of
     * 4.
     */
    population evolve(population pop) const
    {
        // We store some useful variables
        const auto &prob = pop.get_problem(); // This is a const reference, so using set_seed for example will not be
                                              // allowed
        auto dim = prob.get_nx();             // This getter does not return a const reference but a copy
        auto NP = pop.size();

        const auto bounds = prob.get_bounds();
        const auto &lb = bounds.first;
        const auto &ub = bounds.second;

        //Note that the number of equality and inequality constraints has to be set up manually in the problem definition,
        //otherwise PaGMO assumes that there aren't any
        auto NIC = prob.get_nic(); //number of inequality constraints
        auto NEC = prob.get_nec(); //number of equality constraints

        auto fevals0 = prob.get_fevals(); // discount for the fevals already made
        unsigned count = 1u;          // regulates the screen output

        // PREAMBLE-------------------------------------------------------------------------------------------------
        // We start by checking that the problem is suitable for this
        // particular algorithm.

        if (!NP) {
            pagmo_throw(std::invalid_argument, get_name() + " cannot work on an empty population");
        }
        if (prob.is_stochastic()) {
            pagmo_throw(std::invalid_argument,
                        "The problem appears to be stochastic " + get_name() + " cannot deal with it");
        }
        if (prob.get_nc() != 0u) {
            pagmo_throw(std::invalid_argument, "Non linear constraints detected in " + prob.get_name() + " instance. "
                                                   + get_name() + " cannot deal with them.");
        }
        if (m_gen == 0u){
            return pop;
        }



        // ---------------------------------------------------------------------------------------------------------

        // No throws, all valid: we clear the logs
        m_log.clear();

        // Declarations


        // Main ACO loop over generations:
        for (decltype(m_gen) gen = 1u; gen <= m_gen; gen++) {
            // 0 - Logs and prints (verbosity modes > 1: a line is added every m_verbosity generations)


            // At each generation we make a copy of the population into popnew
            population popnew(pop);

            //In the case the algorithm is multi-objective, a decomposition strategy is applied:
            if ( prob.get_nf() - NIC - prob.get_nec() > 1u ) //there might be ineq and eq constraints, so I exclude them from the count
            {

                //THIS PART HAS NOT BEEN DEFINED YET
            }
            //I otherwise proceed with a single-objective algorithm:
            else
            {
                //In this case, the fitness corresponds to the objective values, I can thus directly
                //compute the penalty function values

                auto X = pop.get_x();
                auto fit = pop.get_f(); //this returns a vector in which objectives, equality and inequality
                                        //constraints are concatenated

                //note that pop.get_x()[n][k] goes through the different individuals of the population (index n) and the number of variables (index k)
                //the number of variables can be easily be deducted from counting the bounds.

                //0 - I still have to initialize and define the SA with the first generation of individuals

                //1 - compute penalty functions

                //I define the vector which will store the penalty function values:
                std::vector<double> penalties(NP);

                //loop over the individuals
                for (decltype(NP) i=0u; i<NP; i++)
                {

                    //here, for the penalty computation, you have to pass the i_th element, and not all of them

                    penalties.push_back( penalty_computation( fit[i], pop ) );

                }

                //2 - update and sort solutions in the SA

                //I create a vector where I will store the positions of the various individuals
                std::vector<int> sort_list(NP);

                //I store a vector where the penalties are sorted:
                std::vector<double> sorted_penalties( penalties );
                std::sort (sorted_penalties.begin(), sorted_penalties.end())

                //I now create a vector where I store the position of the stored values: this will help
                //me to find the corresponding individuals and their objective values, later on
                for (decltype(NP) j=0u; j<NP; j++)
                {
                    int count=0;

                    for (decltype(NP) i=0u; i<NP && count=0; i++)
                    {
                        if (sorted_penalties[j] == penalties[i])
                        {
                            if (j==0)
                            {
                                sort_list.push_back(i);
                                count=1;
                            }

                            else
                            {
                                //with the following piece of code I avoid to store the same position in case that two another element
                                //exist with the same value
                              int count_2=0;
                              for(decltype(sort_list.size()) jj=0u; jj<sort_list.size() && count_2=0; jj++)
                              {
                                  if (sort_list(jj)==i)
                                  {
                                      count_2=1;
                                  }
                              }
                              if (count_2==0)
                              {
                                  sort_list.push_back(i);
                                  count=1;
                              }

                            }

                        }
                    }
                }

                if (gen==1) {
                    //here you have to initialize the solution archive (SA)

                }
                else {
                    update_SA(pop, sorted_penalties, sort_list, SA); //you still have to define the SA (Solution Archive)

                }


                //3 - compute pheromone values

                std::vector <double> omega;
                std::vector <double> sigma;
                pheromone_computation(omega, sigma); //you still have to define the inputs to pass




                //4 - use pheromone values to generate new ants, which will become the future generation's variables
                //here you have to define a probability density function and use a random number generator to produce
                //the new ants from it

                //the pdf has the following form:
                //G_h (t) = sum_{k=1}^{K} omega_{k,h} 1/(sigma_h * sqrt(2*pi)) * exp(- (t-mu_{k,h})^2 / (2*(sigma_h)^2) )
                // I thus have all the elements (which I retrieved from the pheromone_computation function)



            }



        } // end of main ACO loop
        return pop;
    }
    /// Sets the seed
    /**
     * @param seed the seed controlling the algorithm stochastic behaviour
     */
    void set_seed(unsigned seed)
    {
        m_e.seed(seed);
        m_seed = seed;
    };
    /// Gets the seed
    /**
     * @return the seed controlling the algorithm stochastic behaviour
     */
    unsigned get_seed() const
    {
        return m_seed;
    }
    /// Sets the algorithm verbosity
    /**
     * Sets the verbosity level of the screen output and of the
     * log returned by get_log(). \p level can be:
     * - 0: no verbosity
     * - >0: will print and log one line each \p level generations.
     *
     * Example (verbosity 1):
     * @code{.unparsed}
     * Gen:        Fevals:        ideal1:        ideal2:        ideal3:
     *   1              0      0.0257554       0.267768       0.974592
     *   2             52      0.0257554       0.267768       0.908174
     *   3            104      0.0257554       0.124483       0.822804
     *   4            156      0.0130094       0.121889       0.650099
     *   5            208     0.00182705      0.0987425       0.650099
     *   6            260      0.0018169      0.0873995       0.509662
     *   7            312     0.00154273      0.0873995       0.492973
     *   8            364     0.00154273      0.0873995       0.471251
     *   9            416    0.000379582      0.0873995       0.471251
     *  10            468    0.000336743      0.0855247       0.432144
     * @endcode
     * Gen, is the generation number, Fevals the number of function evaluation used. The ideal point of the current
     * population follows cropped to its 5th component.
     *
     * @param level verbosity level
     */
    void set_verbosity(unsigned level)
    {
        m_verbosity = level;
    };
    /// Gets the verbosity level
    /**
     * @return the verbosity level
     */
    unsigned get_verbosity() const
    {
        return m_verbosity;
    }  
    /// Gets the generations
    /**
     * @return the number of generations to evolve for
     */
    unsigned get_gen() const
    {
        return m_gen;
    }
    /// Algorithm name
    /**
     * Returns the name of the algorithm.
     *
     * @return <tt> std::string </tt> containing the algorithm name
     */
    std::string get_name() const
    {
        return "gi_aco_mo:";
    }
    /// Extra informations
    /**
     * Returns extra information on the algorithm.
     *
     * @return an <tt> std::string </tt> containing extra informations on the algorithm
     */
    std::string get_extra_info() const
    {
        std::ostringstream ss;
        stream(ss, "\tGenerations: ", m_gen);
        stream(ss, "\n\tAccuracy parameter: ", m_acc);
        stream(ss, "\n\tObjective stopping criterion: ", m_fstop);
        stream(ss, "\n\tImprovement stopping criterion: ", m_impstop);
        stream(ss, "\n\tEvaluation stopping criterion: ", m_evalstop);
        stream(ss, "\n\tFocus parameter: ", m_focus);
        stream(ss, "\n\tKernel: ", m_ker);
        stream(ss, "\n\tOracle parameter: ", m_oracle);
        stream(ss, "\n\tMax number of non-dominated solutions: ", m_paretomax);
        stream(ss, "\n\tPareto precision: ", m_epsilon);
        stream(ss, "\n\tDistribution index for mutation: ", m_e);
        stream(ss, "\n\tSeed: ", m_seed);
        stream(ss, "\n\tVerbosity: ", m_verbosity);

        return ss.str();
    }
    /// Get log

    const log_type &get_log() const
    {
        return m_log;
    }
    /// Object serialization
    /**
     * This method will save/load \p this into the archive \p ar.
     *
     * @param ar target archive.
     *
     * @throws unspecified any exception thrown by the serialization of the UDP and of primitive types.
     */
    template <typename Archive>
    void serialize(Archive &ar)
    {
        ar(m_gen, m_acc, m_fstop, m_impstop, m_evalstop, m_focus, m_ker, m_oracle, m_paretomax, m_epsilon, m_e, m_seed, m_verbosity, m_log);
    }

private:

    //here I define my penalty function computation using oracle penalty method
    double penalty_computation ( const vector_double &f, const population &pop )
    {

        const auto &prob = pop.get_problem();
        auto nec = prob.get_nec();
        //auto nic = prob.get_nic();
        auto nfunc = prob.get_nf() - nec - nic;


        //remember that the get_f() returns the objective vector, equality constraints vector, and inequality
        //constraints vector in the order that I just mentioned.

        double max_ec = f[nfunc];
        double min_ic = f[nfunc+nec];
        double ec_sum_1 = 0;
        double ic_sum_1 = 0;
        double ec_sum_2 = 0;
        double ic_sum_2 = 0;
        //I compute the sum over the equality and inequality constraints (to be used for the residual computation):
        for ( decltype (i) = nfunc; i < nfunc+nec; i++ )
        {
            ec_sum_1 = ec_sum_1 + std::abs(f[i]);
            ec_sum_2 = ec_sum_2 + std::pow(std::abs(f[i]),2);
            if ( i>nfunc && max_ec<f[i] )
            {
                max_ec = f[i];
            }
        }

        for ( decltype (j) = nfunc+nec; j < prob.get_nf(); j++ )
        {
            ic_sum_1 = ic_sum_1 + std::min(std::abs(f[j]),0);
            ic_sum_2 = ic_sum_2 + std::pow(std::min(std::abs(f[j]),0),2);
            if ( j> nfunc+nec && min_ic>f[j] )
            {
                min_ic=f[j];
            }
        }

        unsigned L=2; //if L=1 --> it computes the L_1 norm,
                          //if L=2 --> it computes the L_2 norm,
                          //if L=3 --> it computes the L_inf norm

        //The computation of the residual function is executed:
        double res = 0;

        if( L==1 ) {

            res = ec_sum_1 - ic_sum_1;

        } else if ( L==2 ) {

            res = std::sqrt(ec_sum_2 + ic_sum_2);

        } else {

            res = std::max(max_ec,min_ic);
        }


        //Before computing the penalty function, I need the alpha parameter:

        //I STILL DO NOT DECLARE THE fit VALUE, BECAUSE IT WILL DEPEND ALSO ON THE
        //MULTI-OBJECTIVE PART HOW TO DEFINE IT

        double alpha=0;
        double diff = std::abs(fitness-m_oracle); //I define this value which I will use often

        if ( fitness<=m_oracle ){
            //In this case, I keep the value of alpha = 0
        }

        else if( fitness>m_oracle && res<diff/3.0 ) {

            alpha = (diff* (6.0*std::sqrt(3.0)-2.0)/(6.0*std::sqrt(3)) - res) / (diff-res);

        } else if (fitness>m_oracle && res>=diff/3.0 && res<=diff) {

            alpha = 1.0 - 1.0/(2.0*std::sqrt(diff/res));

        }
        else{ //i.e., fitness>m_oracle && res>diff

            alpha = 1.0/2.0*std::sqrt(diff/res);
        }

        //I now have all the elements to compute the penalty function value:
        double penalty;

        if( fitness>m_oracle && res<diff/3.0 ){

            penalty = alpha*diff + (1-alpha)*res;
        }
        else{
            penalty=-diff;
        }

        return penalty;

    }

    //in the next function I need: 1) the number of the population --> which I will call n_con, 2) the solution archive (where I should place the variables, the obj function
    // the constraints violations and the penalty function values inside). The size of the SA is assumed to be defined as K
    void pheromone_computation( std::vector <double> &OMEGA, std::vector <double> &SIGMA  ) //the size of OMEGA is K, of SIGMA is n_con
    {

        //retrieve K and n_con from the sizes of SA

        //I compute omega (first pheromone value):
        //I declare the omega vector for storing the omega values:


        double omega;

        std::vector<double> J(K) ; // vector with K doubles
        std::iota (std::begin(J), std::end(J), 1); // Fill with 1,2,3,...,K

        double sum = std::accumulate(J.begin(), J.end(),0);

        for ( int k=0; k<K; k++ ){

             omega = ( K-k+1.0 )/(sum);
             OMEGA.push_back(omega);

        }



        //I compute sigma (second pheromone value):


        for ( int h = 0; h < n_con; h++ ){

            //I declare and define D_min and D_max:
            double D_min = std::abs( SA[0][h]-SA[1][h] ); //at first I define D_min using the subtraction of the first two individuals of
                                                          //the same variable stored in the SA --> the index are specified to clarify this
                                                          //but I still have to pass SA to the function
            std::vector <double> D_MIN(n_con);

            double D_max = std::abs( SA[0][h]-SA[1][h] );
            std::vector <double> D_MAX(n_con);


            //I loop over the various individuals of the variable:
            for ( int count=1; count<K-1; count++ ){

                //I confront each individual with the following ones (until all the comparisons are executed):
                for ( int k = count+1; k<K; k++ ){

                    //I update D_min
                    if ( std::abs( SA[count][h]-SA[k][h] )<D_min ){

                        D_min = std::abs( SA[count][h]-SA[k][h]);
                    }

                    //I update D_max
                    if ( std::abs( SA[count][h]-SA[k][h])>D_max ){

                        D_max = std::abs( SA[count][h]-SA[k][h]);
                    }
                }

            }

            D_MIN.push_back( D_min );
            D_MAX.push_back( D_max );

            SIGMA.push_back( (D_max-D_min)/get_gen() );

        }




    }

    void update_SA(const population &pop, std::vector<double> &sorted_vector, std::vector<int> &sorted_list, std::vector< std::vector <double> > &Solution_Archive, )
    {
        //sorted_vector contains the penalties sorted (relative to the generation in which we currently are)
        //sorted_list contains the position values of these penalties wrt their original position as it appears in get_x()
        auto variables = pop.get_x();
        auto objectives = pop.get_f();

        //note that pop.get_x()[n][k] goes through the different individuals of the population (index n) and the number of variables (index k)
        //the number of variables can be easily be deducted from counting the bounds.

        //I now re-order the variables and objective vectors (remember that the objective vector also contains the eq and ineq constraints):
        for (decltype(sorted_list.size()) i=0u; i<sorted_list.size(); i++) {
            variables[i] = pop.get_x()[ sorted_list[i] ];
            objectives[i] = pop.get_f()[ sorted_list[i] ];
        }

        //now I have the individuals sorted in such a way that the first one is the best of its generation, and the last one the worst
        //I can thus compare these individuals with the SA: if for instance the first individual of the sorted generation is worse than
        //the last individual of the SA, then all the others will also be worse, and I can thus interrupt the update. The same holds for
        //the following elements

        //I assume that SA has NP rows and n_con columns: so as many rows as the number of individuals and as many columns as the number
        //of variables for each individual



        int count_2=1;
        for( decltype(SA.size()) j=Solution_Archive.size()-1; j>=0 && count_2==1; j-- )
        {
            count_2=0;
            int count=0;
                for (decltype(sorted_list.size()) i=0u; i<sorted_list.size() && count==0; i++)
                {
                    if (sorted_vector[i] <= Solution_Archive[j][k]) //you have to substitute k with the position in which you will place the penalty
                                                  //function value of the variables in SA
                    {
                        Solution_Archive[j]=variables(i);
                        count_2=1; //if count_2 remains equal to zero, then no values in the sorted vector is found that is better than SA
                    }
                    else
                    {
                        count=1;
                    }
                }
        }



    }

    unsigned m_gen;
    double m_acc;
    int m_fstop;
    int m_impstop;
    int m_evalstop;
    double m_focus;
    int m_ker;
    double m_oracle;
    int m_paretomax;
    double m_epsilon;
    mutable detail::random_engine_type m_e;
    unsigned m_seed;
    unsigned m_verbosity;
    mutable log_type m_log;  
};

} // namespace pagmo

PAGMO_REGISTER_ALGORITHM(pagmo::gi_aco_mo)

#endif
