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

#ifndef PAGMO_ALGORITHMS_GACO_HPP
#define PAGMO_ALGORITHMS_GACO_HPP

#include <algorithm> // std::shuffle, std::transform
#include <iomanip>
#include <numeric> // std::iota, std::inner_product
#include <random>
#include <string>
#include <tuple>
#include <sstream> //std::osstringstream
#include <utility> //std::swap
#include <vector>

#include <pagmo/algorithm.hpp> // needed for the cereal macro
#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/utils/generic.hpp>
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
class g_aco
{
public:
    /// Single entry of the log (gen, fevals, ideal_point)
    typedef std::tuple<unsigned, double, double, double, double, double, double, double, double, double> log_line_type;
    /// The log
    typedef std::vector<log_line_type> log_type;

    /// Constructors
    /**
    * Empty constructor
    * */

    // g_aco( ){ }

    /**
    * Constructs the ACO user defined algorithm for multi-objective optimization.
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
    * @throws std::invalid_argument if \p acc is not \f$ \in [0,1[\f$, \p impstop is not a
    * positive integer, \p evalstop is not a positive integer, \p focus is not \f$ \in [0,1[\f$, \p ants is not a positive integer,
    * \p ker is not a positive integer, \p oracle is not positive, \p paretomax is not a positive integer, \p epsilon is not \f$ \in [0,1[\f$
    */
    g_aco(unsigned gen = 100u, double acc = 0.95, double  fstop = 0.0000001, int impstop = 100, int evalstop = 100,
          double focus = 0,  unsigned ker = 600, double oracle=10000, unsigned paretomax = 10,
            double epsilon = 0.9, unsigned seed = pagmo::random_device::next())
        : m_gen(gen), m_acc(acc), m_fstop(fstop), m_impstop(impstop), m_evalstop(evalstop), m_focus(focus),
          m_ker(ker), m_oracle(oracle), m_paretomax(paretomax), m_epsilon(epsilon), m_e(seed), m_seed(seed), m_verbosity(0u),
          m_log(), res()
    {
        if (acc >= 1. || acc < 0.) {
            pagmo_throw(std::invalid_argument, "The accuracy parameter must be in the [0,1[ range, while a value of "
                                                   + std::to_string(acc) + " was detected");
        }
        if ( focus < 0. ) {
            pagmo_throw(std::invalid_argument,
                        "The focus parameter must be >=0  while a value of "
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

    // it used to be: ;

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
        auto NIC = prob.get_nic();        //number of inequality constraints
        auto NEC = prob.get_nec();        //number of equality constraints
        auto NOBJ = prob.get_nobj();      //number of objectives
        auto NALL = prob.get_nf();        // NALL=NOBJ+NEC+NIC

        auto fevals0 = prob.get_fevals(); // discount for the fevals already made
        unsigned count_screen = 1u;          // regulates the screen output


        std::vector< vector_double > SA( m_ker, vector_double(1+dim+NALL,1) );

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
        //I verify that the solution archive is smaller or equal than the population size
        if (m_ker > NP){
            pagmo_throw(std::invalid_argument, get_name() + " cannot work with a solution archive bigger than the population size");

        }
        // ---------------------------------------------------------------------------------------------------------

        // No throws, all valid: we clear the logs
        m_log.clear();

        //0 - I initialize and define the SA with the first generation of individuals

        //I store in the first column the penalty values, in the following columns the variables, the objectives values
        //the inequality constraints values and the equality constraints values. The number of rows of SA i


        // Main ACO loop over generations:
        for (decltype(m_gen) gen = 1u; gen <= m_gen; gen++ ) {

            //At each generation we make a copy of the population into popnew
            population popnew(pop);

            //I define the variables which will count the runs without improvements and the function evaluations:
            int count_impstop = 0;
            int count_evalstop = 0;

            //In the case the algorithm is multi-objective, a decomposition strategy is applied:
            if ( prob.get_nobj() > 1u ) {
                //THIS PART HAS NOT BEEN DEFINED YET
            }

            //I otherwise proceed with a single-objective algorithm:
            else
            {


                auto X = pop.get_x();
                //std::cout << "Individuals (X and Y):" << std::endl;
                //for (int MM=0; MM<NP && gen==1; ++MM){
                    //std::cout <<  X[MM][0] << std::endl;
                    //std::cout <<  X[MM][1] << std::endl;
                //}
                //std::cout <<  X[0][0] << std::endl;
                //std::cout <<  X[0][1] << std::endl;
                //std::cout << X[0].size() << std::endl;
                //The following returns a vector of vectors in which objectives, equality and inequality constraints are concatenated,for each individual
                auto fit = pop.get_f();

                //std::cout << "fit retrieved" << std::endl;
                //for (int ik=0u; ik<NP; ++ik){
                    //std::cout << fit[ik][0] << std::endl;
                //}


                //std::cout << "First Obj:" << std::endl;
                //std::cout << fit[0][0] << std::endl;

                //std::cout << fit[0].size() << std::endl;


                //note that pop.get_x()[n][k] goes through the different individuals of the population (index n) and the number of variables (index k)
                //the number of variables can be easily be deducted from counting the bounds.


                //I verify whether the maximum number of function evaluations or improvements has been exceeded, if yes I return the population and interrupt the algorithm
                if ( m_impstop!=0  && count_impstop>=m_impstop){
                    return pop;
                }


                if ( m_evalstop!=0 && count_evalstop>=m_evalstop ){
                    return pop;
                }



                //1 - compute penalty functions

                //I define the vector which will store the penalty function values:
                vector_double penalties;
                int feasible_set=0;





                //std::cout << "penalties:" << std::endl;
                for ( decltype(NP) i=0u; i<NP; ++i )
                {


                    //I first verify whether there is a solution that is smaller or equal the fstop parameter, in the case that it is different than zero
                    if ( m_fstop!=0 && fit[i][0]<=m_fstop ){
                        std::cout << "if a value of zero is desired as fstop, please insert a very small value instead (e.g. 0.0000001)" << std::endl;
                        return pop;
                    }



                    int T=0;
                    int T_2=0;


                    //I verify that the equality and inequality constraints make the solutions feasible, if not, they are discarded:
                    for ( decltype(NEC) i_nec=1u; i_nec<=NEC && T==0; ++i_nec ){
                        if( std::abs(fit[i][i_nec]) > m_acc )
                        {
                            T=1;
                        }
                    }
                    for ( decltype(NIC) i_nic=1u; i_nic<=NIC && T_2==0; ++i_nic ){
                        if( fit[i][i_nic] >= -m_acc ){ //remember that the ineq constraints are of the kind: g_ineq(X)>=0
                            T_2=1;

                        }
                    }
                    if (T==0 & T_2==0) {

                        //Penalty computation is here executed:
                        penalties.push_back( penalty_computation( fit[i], pop ) );

                        //std::cout << "penalty:" << std::endl;
                        //std::cout << penalty_computation(fit[i],pop) << std::endl;
                        //std::cout << "obj:" << std::endl;
                        //std::cout << fit[i][0] << std::endl;
                        ++feasible_set;

                    }
                    else{
                        //where this is not true, I store an infinity
                        penalties.push_back( std::numeric_limits<double>::infinity() );
                    }

                    //std::cout << penalties[i] << std::endl;

                }

                if (feasible_set>=1){
                    //loop over the individuals

                    //2 - update and sort solutions in the SA (only the feasible ones)

                    //std::cout << "penalty size:"<< std::endl;
                    //std::cout << penalties.size() << std::endl;
                    //std::cout << "1st penalty value:"<< std::endl;
                    //std::cout << penalties[0] << std::endl;
                    //std::cout << "900th penalty value:"<< std::endl;
                    //std::cout << penalties[900] << std::endl;
                    //I create a vector where I will store the positions of the various individuals
                    vector_double sort_list;

                    //I store a vector where the penalties are sorted:
                    vector_double sorted_penalties( penalties );

                    std::sort (sorted_penalties.begin(), sorted_penalties.end());

                    //std::cout << "first sorted penalty" << std::endl;
                    //std::cout << sorted_penalties[0] << std::endl;
                    //std::cout << "last sorted penalty" << std::endl;
                    //std::cout << sorted_penalties[999] << std::endl;

                    //I now create a vector where I store the position of the stored values: this will help
                    //me to find the corresponding individuals and their objective values, later on
                    //std::cout << "PENALTIES STORED" << std::endl;
                    for ( decltype(penalties.size()) j=0u; j<penalties.size(); ++j ) {
                        int count=0;

                        for ( decltype(penalties.size()) i=0u; i<penalties.size() && count==0; ++i ) {
                            if (sorted_penalties[j] == penalties[i] && sorted_penalties[j]!=std::numeric_limits<double>::infinity() ) {
                                if (j==0) {
                                    sort_list.push_back(i);
                                    count=1;
                                }

                                else {
                                    //with the following piece of code I avoid to store the same position in case that two another element
                                    //exist with the same value
                                    int count_2=0;
                                    for(decltype(sort_list.size()) jj=0u; jj<sort_list.size() && count_2==0; ++jj) {
                                        if (sort_list[jj]==i)
                                        {
                                            count_2=1;
                                        }
                                    }
                                    if (count_2==0) {
                                        sort_list.push_back(i);
                                        count=1;
                                    }

                                }


                            }

                        }
                        //std::cout << sorted_penalties[j] << std::endl;
                    }



                        if (gen==1) {

                            if ( feasible_set<m_ker ){
                                pagmo_throw(std::invalid_argument,
                                            " Error: the initial population does not have at least m_ker feasible individuals to be stored in the solution archive ");

                            }


                            //I initialize the solution archive (SA): in order to do this I store the vectors generated in the first generation
                            //by taking into account their penalty values. In this way, the first vector in the SA (i.e., the first row in which
                            //penalty value, variables, objective functions values, equality constraints values, inequality constraints values are
                            //stored) represents the best one (i.e., the one that has the smallest penalty function value among the individuals of
                            //that generation), whereas the last vector represents the worst.

                            for (int i=0u; i<m_ker; ++i){
                                //std::cout << "New Individual" << std::endl;
                                //std::cout << "Oracle:" << std::endl;
                                //std::cout << m_oracle << std::endl;

                                //std::cout << "Penalty:" << std::endl;
                                SA[i][0] = penalties[sort_list[i]];
                                //std::cout << SA[i][0] << std::endl;

                                //std::cout << "Variables:" << std::endl;
                                for (decltype(dim) J=0; J<dim; ++J){
                                    SA[i][J+1] = X[sort_list[i]][J];
                                    //std::cout << SA[i][J+1] << std::endl;

                                }
                                //std::cout << "Objectives" << std::endl;
                                for (decltype(NALL) J=0; J<NALL; ++J){
                                    SA[i][J+1+dim] = fit[sort_list[i]][J];
                                    //std::cout << fit[sort_list[i]][J] << std::endl;
                                    //std::cout << i << std::endl;
                                    //std::cout << SA[i][0] << std::endl;



                                }


                            }


                            if (m_impstop!=0){
                                ++count_impstop;
                            }



                        }
                        else{

                            //std::cout << "gen>1" << std::endl;
                            update_SA(pop, sorted_penalties, sort_list, count_impstop, count_evalstop, SA);


                        }


                        // 0 - Logs and prints (verbosity modes > 1: a line is added every m_verbosity generations)
                        if (m_verbosity > 0u) {
                            // Every m_verbosity generations print a log line
                            if (gen % m_verbosity == 1u || m_verbosity == 1u) {
                                auto best_penalty = SA[0][0];
                                auto best_fit = SA[0][1+dim];
                                auto worst_penalty = SA[m_ker-1][0];
                                auto worst_fit = SA[m_ker-1][1+dim];
                                auto best_x = SA[0][1];
                                auto best_y = SA[0][2];
                                double dx = 0., df=0.;
                                // The population flattness in chromosome
                                for (decltype(dim) i = 0u; i < dim; ++i) {
                                    dx += std::abs(SA[m_ker-1][1+i] - SA[0][1+i]);
                                }
                                // The population flattness in fitness
                                df = std::abs(SA[m_ker-1][1+dim] - SA[0][1+dim]);
                                // Every 50 lines print the column names
                                if ( count_screen>0 ){ //count_screen % 50u == 1u) {
                                    print("\n", std::setw(7), "Gen:", std::setw(15),
                                          "Best:", std::setw(15), "Best penalty:", std::setw(15), "Worst:", std::setw(15),
                                          "Best X:", std::setw(15), "Best Y:", std::setw(15),
                                          "Worst penalty:", std::setw(15), "Oracle:", std::setw(15),
                                          "dx:", std::setw(15), std::setw(15), "df:", '\n');
                                }
                                print(std::setw(7), gen, std::setw(15),
                                      best_fit, std::setw(15), best_penalty, std::setw(15),
                                      worst_fit, std::setw(15), best_x, std::setw(15), best_y, std::setw(15),
                                      worst_penalty, std::setw(15), m_oracle, std::setw(15),
                                      dx, std::setw(15), df, '\n');
                                ++count_screen;
                                // Logs
                                m_log.emplace_back(gen, best_fit, best_penalty, worst_fit, worst_penalty, m_oracle, dx, df);
                            }
                        }

                        //std::cout << SA[0][dim+1] << std::endl;



                        //3 - compute pheromone values

                        vector_double omega;
                        vector_double sigma;

                        pheromone_computation( omega, sigma, pop, gen, SA );

                        //std::cout << "size of SIGMA" << std::endl;
                        //std::cout << sigma.size() << std::endl;
                        //std::cout << sigma[0] << std::endl;
                        //std::cout << sigma[1]<< std::endl;

                        //4 - use pheromone values to generate new ants, which will become the future generation's variables
                        //here you have to define a probability density function and use a random number generator to produce
                        //the new ants from it

                        //I create the vector of vectors where I will store all the new ants (i.e., individuals) which will be generated
                        std::vector < vector_double > new_ants;

                        generate_new_ants( popnew, omega, sigma, dim, new_ants, NP, gen, SA );

                        if (gen==1 || gen==5 || gen==9){

                        for (int j=0; j<new_ants.size(); ++j){

                            //std::cout << "X"<< std::endl;
                            //std::cout << new_ants[j][0] << std::endl;
                            //std::cout << "Y" << std::endl;
                            //std::cout << new_ants[j][1] << std::endl;
                        }

                        }



                        //std::cout << "fitness set:" << std::endl;
                        for ( population::size_type i=0; i<NP; ++i){
                            vector_double ant;
                            //I compute the fitness for each new individual which was generated in the generated_new_ants(..) function
                            for (decltype(new_ants[i].size()) ii=0u; ii<new_ants[i].size(); ++ii  ){
                                ant.push_back(new_ants[i][ii]);

                            }

                            auto fitness = prob.fitness(ant);

                            //std::cout << fitness[0] << std::endl;


                            //I save the individuals for the next generation
                            pop.set_xf(i, ant, fitness);


                        }

                        //the oracle parameter is updated after each optimization run:
                        if ( SA[0][1+dim]<m_oracle && res==0 ){
                            m_oracle = SA[0][1+dim];
                            if (gen>0){
                            //std::cout << "CAMBIATO oracle:" << std::endl;
                            //std::cout << m_oracle << std::endl;
                            //std::cout << "SA best:" << std::endl;
                            //std::cout << SA[0][1+dim] << std::endl;
                            //std::cout << "Secondo:" << std::endl;
                            //std::cout << penalty << std::endl;
                            //I can now compute the penalty function value
                        }
                            //std::cout << "PENALTY NEW:" << std::endl;
                            for (int rows=0u; rows<m_ker; ++rows){
                                double FITNESS = SA[rows][1+dim];
                                double alpha=0;
                                double diff = std::abs(FITNESS-m_oracle); //I define this value which I will use often

                                if( FITNESS>m_oracle && res<diff/3.0 ) {
                                    alpha = (diff* (6.0*std::sqrt(3.0)-2.0)/(6.0*std::sqrt(3)) - res) / (diff-res);

                                } else if (FITNESS>m_oracle && res>=diff/3.0 && res<=diff) {

                                    alpha = 1.0 - 1.0/( 2.0*std::sqrt(diff/res) );

                                }
                                else if ( FITNESS>m_oracle && res>diff ) {

                                    alpha = 1.0/2.0*std::sqrt(diff/res);
                                }


                                //I can now compute the penalty function value
                                if( FITNESS>m_oracle || res>0 ){
                                     SA[rows][0] = alpha*diff + (1-alpha)*res;

                                }
                                else if (FITNESS<=m_oracle && res==0){
                                    SA[rows][0] = -diff;
                                }
                               // std::cout << SA[rows][0] << std::endl;
                            }


                        }


                        if (gen==1 || gen==2 || gen==3){
                            //std::cout << "Solution archive variables" << std::endl;
                            for(decltype(m_ker) M=0u; M<m_ker; ++M){
                               // std::cout << SA[M][1] << std::endl;
                                //std::cout << SA[M][2] << std::endl;
                            }
                        }



                       // std::cout << "SA fitness:"<< std::endl;
                        for (decltype(m_ker) KK=0u; KK<m_ker; ++KK){
                            //std::cout << SA[KK][1+dim] << std::endl;
                        }



                    }
                    else{
                        pagmo_throw(std::invalid_argument,
                                    " Error: the population does not have any feasible individuals to be compared with the solution archive  ");

                    }


                }


            }// end of main ACO loop

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
        return "g_aco:";
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
        ar(m_gen, m_acc, m_fstop, m_impstop, m_evalstop, m_focus, m_ker, m_oracle, m_paretomax, m_epsilon, m_e, m_seed, m_verbosity, m_log, res, SA);
    }

        std::vector< vector_double > SA;
private:

    //here I define my penalty function computation using oracle penalty method
    double penalty_computation ( const vector_double &f, const population &pop ) const
    {
        /**
        * Function which computes the penalty function values for each individual of the population
        *
        * @param[in] f Fitness values: vector in which the objective functions values, equality constraints and inequality constraints are stored for each passed individual
        * @param[in] pop Population: the current population is passed
        * @param[in] SA Solution archive: the solution archive is useful for retrieving the current individuals (which will be the means of the new pdf)
        */

        const auto &prob = pop.get_problem();
        auto nec = prob.get_nec();
        auto nic = prob.get_nic();
        auto nfunc = prob.get_nobj();


        //remember that the get_f() returns the objective vector, equality constraints vector, and inequality
        //constraints vector in the order that I just mentioned.

        double max_ec = f[nfunc];
        double min_ic = f[nfunc+nec];
        double ec_sum_1 = 0;
        double ic_sum_1 = 0;
        double ec_sum_2 = 0;
        double ic_sum_2 = 0;

        //The residual function variable is assigned (it will be used inside the penalty_computation function):
        res=0;

        if ( nic!=0 || nec!=0 || nfunc>1 ){
            //I compute the sum over the equality and inequality constraints (to be used for the residual computation):
            for ( decltype (nfunc+nec) i = nfunc; i < nfunc+nec; ++i )
            {
                ec_sum_1 = ec_sum_1 + std::abs(f[i]);
                ec_sum_2 = ec_sum_2 + std::pow(std::abs(f[i]),2);
                if ( i>nfunc && max_ec<f[i] ) {
                    max_ec = f[i];
                }
            }

            for ( decltype ( prob.get_nf() ) j = nfunc+nec; j < prob.get_nf(); ++j ) {
                ic_sum_1 = ic_sum_1 + std::min(std::abs(f[j]),0.0);
                ic_sum_2 = ic_sum_2 + std::pow(std::min(std::abs(f[j]),0.0),2);

                if ( j> nfunc+nec && min_ic>f[j] ) {
                    min_ic=f[j];
                }
            }

            unsigned L=2;     //if L=1 --> it computes the L_1 norm,
            //if L=2 --> it computes the L_2 norm,
            //if L=3 --> it computes the L_inf norm



            if( L==1 ) {
                res = ec_sum_1 - ic_sum_1;

            } else if ( L==2 ) {
                res = std::sqrt(ec_sum_2 + ic_sum_2);

            } else {
                res = std::max(max_ec,min_ic);
            }
        }

        //Before computing the penalty function, I need the alpha parameter:

        //I STILL DO NOT DECLARE THE 'fitness' VALUE, BECAUSE IT WILL DEPEND ALSO ON THE
        //MULTI-OBJECTIVE PART HOW TO DEFINE IT

        //for single objective, for now, is enough to do:
        auto fitness = f[0];

        double alpha=0;
        double diff = std::abs(fitness-m_oracle); //I define this value which I will use often
        //I define the penalty function value variable
        double penalty;


        if( fitness>m_oracle && res<diff/3.0 ) {
            alpha = (diff* (6.0*std::sqrt(3.0)-2.0)/(6.0*std::sqrt(3)) - res) / (diff-res);

        } else if (fitness>m_oracle && res>=diff/3.0 && res<=diff) {

            alpha = 1.0 - 1.0/( 2.0*std::sqrt(diff/res) );

        }
        else if ( fitness>m_oracle && res>diff ) {

            alpha = 1.0/2.0*std::sqrt(diff/res);
        }


        //I can now compute the penalty function value
        if( fitness>m_oracle || res>0 ){
            penalty = alpha*diff + (1-alpha)*res;

        }
        else if (fitness<=m_oracle && res==0){
            penalty = -diff;
        }


        return penalty;


    }

    //in the next function I need: 1) the number of the population --> which I will call n_con, 2) the solution archive (where I should place the variables, the obj function
    // the constraints violations and the penalty function values inside). The size of the SA is assumed to be defined as K
    void pheromone_computation( vector_double &OMEGA, vector_double &SIGMA, const population &popul, const unsigned generations, std::vector< vector_double > &SA  ) const //the size of OMEGA is K, of SIGMA is n_con
    {

        /**
        * Function which computes the pheromone values (useful for generating offspring)
        *
        * @param[in] OMEGA Omega: the weights are passed to be modified (i.e., they are one of the pheromone values)
        * @param[in] SIGMA Sigma: the standard deviations are passed to be modified (i.e., they are one of the pheromone values)
        * @param[in] popul Population: the current population is passed
        * @param[in] SA Solution archive: the solution archive is useful for retrieving the current individuals (which will be the means of the new pdf)
        */

        const auto &prob = popul.get_problem();
        const auto bounds = prob.get_bounds();
        const auto &lb = bounds.first;
        const auto &ub = bounds.second;
        auto n_con = prob.get_nx();

        //retrieve K and n_con from the sizes of SA

        //I compute omega (first pheromone value):
        //I declare the omega vector for storing the omega values:


        double omega;

        vector_double J(m_ker) ; // vector with 'ker' doubles
        std::iota (std::begin(J), std::end(J), 1); //Fill with 1,2,3,...,K

        double sum = std::accumulate(J.begin(), J.end(),0); //Do the sum of the elements
        //std::cout << sum << std::endl;


        //I compute sigma (second pheromone value):

        for ( decltype(m_ker) k=1; k<=m_ker; ++k ){
             omega = ( m_ker-k+1.0 )/(sum);
             OMEGA.push_back(omega);

        }





        for ( decltype(n_con) h = 1; h <= n_con; ++h ){

            //I declare and define D_min and D_max:
            //at first I define D_min using the subtraction of the first two individuals of the same variable stored in the SA
            double D_min = std::abs( SA[0][h]-SA[1][h] );
            vector_double D_MIN;

            double D_max = std::abs( SA[0][h]-SA[1][h] );
            vector_double D_MAX;


            //I loop over the various individuals of the variable:
            for ( decltype(m_ker) count=0; count<m_ker-1.0; ++count ){

                //I confront each individual with the following ones (until all the comparisons are executed):
                for ( decltype(m_ker)  k = count+1; k<m_ker; ++k ){

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

            //std::cout << "gen" << std::endl;
            //std::cout << generations << std::endl;
            if (generations == 1){
                //std::cout << "D_min and D_max" << std::endl;
                //std::cout << D_min << std::endl;
                //std::cout << D_max << std::endl;
            }
            D_MIN.push_back( D_min );
            D_MAX.push_back( D_max );

            //std::cout << "Size of D_min and D_max (in order):" << std::endl;
            //std::cout << D_MIN.size() << std::endl;
            //std::cout << D_MAX.size() << std::endl;


            if ( m_focus!=0 &&  ( (D_max-D_min)/generations > (ub[h-1]-lb[h-1])/m_focus) ) {
                //In case a value for the focus parameter (different than zero) is passed, this limits the maximum value of the standard deviation
                SIGMA.push_back((ub[h-1]-lb[h-1])/m_focus);

            }

            else{
                SIGMA.push_back( (D_max-D_min)/generations );
            }


        }




    }

    void update_SA(const population &pop, vector_double &sorted_vector, std::vector<double> &sorted_list, int &N_impstop, int &N_evalstop, std::vector< vector_double > &SA) const
    {
        /**
        * Function which updates the solution archive, if better solutions are found
        *
        * @param[in] pop Population: the current population is passed
        * @param[in] stored_vector Stored penalty vector: the vector in which the penalties of the current population are stored from the best to the worst is passed
        * @param[in] stored_list Positions of stored penalties: this represents the positions of the individuals wrt their penalties as they are stored in the stored_vector
        * @param[in] Solution_Archive Solution archive: the solution archive is useful for retrieving the current individuals (which will be the means of the new pdf)
        */

        bool N_impstop_count=false;
        bool N_evalstop_count=false;

        //sorted_vector contains the penalties sorted (relative to the generation in which we currently are)
        //sorted_list contains the position values of these penalties wrt their original position as it appears in get_x()
        auto variables = pop.get_x();
        auto objectives = pop.get_f();


        //note that pop.get_x()[n][k] goes through the different individuals of the population (index n) and the number of variables (index k)
        //the number of variables can be easily be deducted from counting the bounds.

        //I now re-order the variables and objective vectors (remember that the objective vector also contains the eq and ineq constraints):
        for ( decltype(sorted_list.size()) i=0u; i<sorted_list.size(); ++i ) {
            variables[i] = pop.get_x()[ sorted_list[i] ];
            objectives[i] = pop.get_f()[ sorted_list[i] ];
        }

        //now I have the individuals sorted in such a way that the first one is the best of its generation, and the last one the worst
        //I can thus compare these individuals with the SA: if for instance the first individual of the sorted generation is worse than
        //the last individual of the SA, then all the others will also be worse, and I can thus interrupt the update. The same holds for
        //the following elements

        //I assume that SA has NP rows (which you call K in the lit. study) and n_con columns: so as many rows as the number of individuals
        //and as many columns as the number of variables for each individual


        //std::cout << "1st err" << std::endl;
        vector_double saved_values_positions;
        bool condition = false;
        bool count_2 = false;
        bool count = false;
        int j = 0;

        //std::cout << "sorted vector values:"<<std::endl;
        for (decltype(sorted_vector.size()) jj=0; jj<sorted_vector.size(); ++jj){
           // std::cout << sorted_vector[jj] << std::endl;
        }


        for (int i=m_ker-1; i>=0 && count==false; --i){

            //std::cout << "sec err" << std::endl;
            //std::cout << SA[i][0] << std::endl;
            //std::cout << sorted_vector.size() << std::endl;
            //std::cout << "third err" << std::endl;
            //std::cout << i << std::endl;
            if (sorted_vector[j]<SA[i][0]){
                for ( int ii=m_ker-1; ii>=m_ker-j-1 && ii>=0; --ii){
                    //std::cout << "fourth err" << std::endl;
                    //std::cout << j << std::endl;
                    //std::cout << m_ker - 1 << std::endl;
                    //std::cout << m_ker-j-1 << std::endl;
                    //std::cout << ii << std::endl;
                    if (sorted_vector[j]>SA[ii][0]){
                        count_2=true;
                        //std::cout << "fifth err" << std::endl;
                    }

                }
                if (count_2==false){

                    saved_values_positions.push_back(j);
                    //std::cout<<"saved value position & value:" << std::endl;
                    //std::cout << j << std::endl;
                    //std::cout << sorted_vector[j] << std::endl;
                    ++j;
                    condition = true;
                    //std::cout << "Here 1" << std::endl;


                }

            }
            else{
                count = true;
            }

        }



        if(condition==true){
            int i=0;
            //std::cout << "New Sol Archive (penalty, variables, objectives)" << std::endl;
            for (int j=m_ker-1; j>=m_ker-1-saved_values_positions[saved_values_positions.size()-1]; --j){

                //I now store the penalty:
                SA[j][0] = sorted_vector[saved_values_positions[i]];
                //std::cout << SA[j][0] << std::endl;


                //I now store the variables:
                for (decltype(variables[0].size()) i_var=0; i_var<variables[0].size(); ++i_var ){
                    SA[j][1+i_var] = variables[saved_values_positions[i]][i_var];
                    //std::cout << SA[j][1+i_var] << std::endl;
                }

                //I now store the objectives, equality constraints and inequality constraints (in this order):
                for (decltype(objectives[0].size()) i_obj=0; i_obj<objectives[0].size(); ++i_obj ){
                    SA[j][1+variables[0].size()+i_obj] = objectives[saved_values_positions[i]][i_obj];
                    //std::cout << SA[j][1+variables[0].size()+i_obj] << std::endl;
                }
                ++i;

            }
            //std::cout << "4th err" << std::endl;


            //I now have to re-order the SA depending on the penalty functions values:


            //I create a vector where I will store the positions of the various individuals
            vector_double sort_list;

            //I store a vector where the penalties are sorted:
            vector_double new_penalties;

            for (int i=0; i<m_ker; ++i ){
                new_penalties.push_back(SA[i][0]);

            }

            //std::cout << "5th err" << std::endl;

            std::sort (new_penalties.begin(), new_penalties.end());

            //std::cout << "first sorted penalty" << std::endl;
            //std::cout << new_penalties[0] << std::endl;
            //std::cout << "last sorted penalty" << std::endl;
            //std::cout << new_penalties[999] << std::endl;

            //I now create a vector where I store the position of the stored values: this will help
            //me to find the corresponding individuals and their objective values, later on

            //std::cout << "new PEN STORED" << std::endl;
            for ( decltype(new_penalties.size()) j=0u; j<new_penalties.size(); ++j ) {
                int count=0;

                for ( decltype(new_penalties.size()) i=0u; i<new_penalties.size() && count==0; ++i ) {
                    if (new_penalties[j] == SA[i][0] && new_penalties[j]!=std::numeric_limits<double>::infinity() ) {
                        if (j==0) {
                            sort_list.push_back(i);
                            count=1;
                        }

                        else {
                            //with the following piece of code I avoid to store the same position in case that two another element
                            //exist with the same value
                            int count_2=0;
                            for(decltype(sort_list.size()) jj=0u; jj<sort_list.size() && count_2==0; ++jj) {
                                if (sort_list[jj]==i)
                                {
                                    count_2=1;
                                }
                            }
                            if (count_2==0) {
                                sort_list.push_back(i);
                                count=1;
                            }

                        }

                    }

                }
                //std::cout << new_penalties[j] << std::endl;
            }

            //std::cout << "6th err" << std::endl;


            std::vector < vector_double > old_SA (SA);

            for ( decltype(sort_list.size()) j=0u; j<sort_list.size(); ++j){

                //I reorder the penalty:
                SA[j][0] = old_SA[sort_list[j]][0];
                //std::cout << "SIzeof oldSA" << std::endl;
                //std::cout << old_SA.size() << std::endl;
                //std::cout << old_SA[j].size() << std::endl;
                //std::cout << "Penalty old:" << std::endl;
                //std::cout << old_SA[j][0] << std::endl;
                //std::cout << "Penalty new:" << std::endl;
                //std::cout << old_SA[sort_list[j]][0] << std::endl;

                //I reorder the variables:
                //std::cout << "Variabbbles" << std::endl;
                for (decltype(variables[0].size()) i_var=0u; i_var<variables[0].size() ;++i_var){
                    SA[j][1+i_var] = old_SA[sort_list[j]][1+i_var];
                    //std::cout << old_SA[sort_list[j]][1+i_var] << std::endl;

                }
                //std::cout << "7th err" << std::endl;


                //std::cout << "Obbbj" << std::endl;

                //I reorder the objectives, equality and inequality constraints (in this order):
                for (decltype(objectives[0].size()) i_obj=0u; i_obj<objectives[0].size() ;++i_obj){
                    SA[j][1+variables[0].size()+i_obj] = old_SA[sort_list[j]][1+variables[0].size()+i_obj];
                    //std::cout << old_SA[sort_list[j]][1+variables[0].size()+i_obj] << std::endl;


                }
            }
        }


    }

    void generate_new_ants( const population &pop, vector_double omega, vector_double sigma, int n_con, std::vector< vector_double > &X_new, int pop_size, double gen, std::vector< vector_double > &SA ) const
    {
        /**
        * Function which generates new individuals (i.e., ants)
        *
        * @param[in] omega Omega: one of the three pheromone values. These are the weights which are used in the multi-kernel gaussian probability distribution
        * @param[in] sigma Sigma: one of the three pheromone values. These are the standard deviations which are used in the multi-kernel gaussian probability distribution
        * @param[in] SA Solution archive: the solution archive is useful for retrieving the current individuals (which will be the means of the new pdf)
        * @param[in] n_con Number of variables: this represents the number of variables of the problem
        * @param[in] X_new New ants: in this vector the new ants which will be generated are stored
        */

        const auto &prob = pop.get_problem();
        const auto bounds = prob.get_bounds();
        const auto &lb = bounds.first;
        const auto &ub = bounds.second;

        //I hereby generate the new ants based on a multi-kernel gaussian probability density function. This pdf is nothing more than a weighted sum of several gaussian pdf
        //hence, in order to reproduce it, we first compute the numbers generated by the various gaussian pdf (using the mean and standard deviation values), and we then
        //weight them over the weights (omega) --> weights and stddev were computed in the pheromone_computation function, whereas the mean of the various gaussian pdf is
        //directly related to the previous individuals (i.e., indeed, their exact value is assumed to be the mean)


        //X_new is the vector of vectors where I will store all the generated ants






        //the number of individuals which are contained in the solution archive is K, whereas I assume to pass the number of different variables which are to be optimized:
        for ( decltype(pop_size) j=0u; j<pop_size; ++j )
        {

            vector_double X_new_k; //here I store all the variables associated with the k_th element of the SA

            for ( decltype(n_con) h=0u; h<n_con; ++h )
            {
                double g_h=0;

                double  verif=0;
                double l_h;
                for ( decltype(SA.size()) k=0u; k< SA.size(); ++k )
                {
                    // Mersenne twister PRNG
                    std::mt19937 generator(m_seed+gen+h+j);


                    //after you define SA you have to correct this--> in this case I assumed that the SA has as many rows as the number of elements stored in the SA (i.e., K)
                    //and as many columns as the 1 (this first element is useful for placing the penalty function values) +number of variables + number of objectives + number of iec
                    //+ number of ec which are defined in the problem --> for accessing the variables, I thus have to go from 1 (i.e., second position) to n_con (number of variables stored
                    //inside the SA)
                    //std::cout << "individual number" << std::endl;
                    //std::cout << k << std::endl;
                    //std::cout << "SA size" << std::endl;
                    //std::cout << SA.size() << std::endl;
                    //std::cout << "mean" << std::endl;
                    //std::cout << SA[k][1+h] << std::endl;
                    //std::cout << "stddev" << std::endl;
                    //std::cout << sigma[h] << std::endl;
                    //std::cout << "weight" << std::endl;
                    //std::cout << omega[k] << std::endl;



                    std::normal_distribution <double> gauss_pdf {SA[k][1+h], sigma[h]};
                    double var =gauss_pdf(generator);
                    if (gen==1){
                    //std::cout<<"generator"<<std::endl;

                    //std::cout << var <<std::endl;
                    //std::cout << omega[k] << std::endl;
                    }

                    l_h=omega[k]*var;

                    //std::cout << "L_H" << std::endl;
                    //std::cout << l_h << std::endl;
                    verif += omega[k];
                    g_h += l_h;
                    if (h==0 && gen==1){
                        //std::cout << "g_h_x:" << std::endl;
                        //std::cout << l_h << std::endl;
                        //std::cout << "g_h_sum" << std::endl;
                        //std::cout << g_h << std::endl;
                    }
                    //std::cout << "g_h (start)" << std::endl;
                    //std::cout << g_h << std::endl;
                    //std::cout << "g_h (end)" << std::endl;
                    //the pdf has the following form:
                    //G_h (t) = sum_{k=1}^{K} omega_{k,h} 1/(sigma_h * sqrt(2*pi)) * exp(- (t-mu_{k,h})^2 / (2*(sigma_h)^2) )
                    // I thus have all the elements to compute it (which I retrieved from the pheromone_computation function)

                }


                if(g_h<lb[h] || g_h>ub[h]){
                     std::cout << "Lower or upper bounds violated, the ant is replaced:" << std::endl;

                     std::cout << g_h << std::endl;
                     while (g_h<lb[h] || g_h>ub[h]){
                         std::cout << "IMPALLATO" << std::endl;
                         g_h=0;
                         double index=pagmo::random_device::next();

                         for ( decltype(SA.size()) k=0u; k< SA.size(); ++k )
                         {
                             // Mersenne twister PRNG
                             std::mt19937 generator(m_seed+gen+h+j+index);
                             std::normal_distribution <double> gauss_pdf {SA[k][1+h], sigma[h]};
                             double var =gauss_pdf(generator);
                             l_h=omega[k]*var;
                             g_h += l_h;
                         }
                     }
                     std::cout<<"BOUNDS"<<std::endl;
                     std::cout<<lb[h]<<std::endl;
                     std::cout<<ub[h]<<std::endl;
                     std::cout<<"NEW"<<std::endl;
                     std::cout<<g_h<<std::endl;
                }

                X_new_k.push_back( g_h );

                //std::cout << "NEW ANT_2" << std::endl;
                //std::cout << X_new_k[1] << std::endl;

            }

            if (gen==1 || gen==2){
            //std::cout << "NEW ANT X" << std::endl;
            //std::cout << X_new_k[0] << std::endl;
            //std::cout << "NEW ANT Y" << std::endl;
            //std::cout << X_new_k[1] << std::endl;
            }
            X_new.push_back(X_new_k);

        }


    }

    unsigned m_gen;
    double m_acc;
    double m_fstop;
    int m_impstop;
    int m_evalstop;
    double m_focus;
    unsigned m_ker;
    mutable double m_oracle;
    unsigned m_paretomax;
    double m_epsilon;
    mutable detail::random_engine_type m_e;
    unsigned m_seed;
    unsigned m_verbosity;
    mutable log_type m_log;
    mutable double res;



};

} // namespace pagmo

PAGMO_REGISTER_ALGORITHM(pagmo::g_aco)

#endif
