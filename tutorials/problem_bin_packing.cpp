#include <cmath>
#include <initializer_list>
#include <iostream>
#include <utility>

#include <pagmo/problem.hpp>
#include <pagmo/types.hpp>
#include <pagmo/algorithm.hpp>
#include <pagmo/population.hpp>
#include <pagmo/algorithms/gaco.hpp>
#include <pagmo/archipelago.hpp>
#include <pagmo/topologies/fully_connected.hpp>
#include <pagmo/algorithms/ihs.hpp>

struct bin_packing {
    // Number of equality constraints.
    pagmo::vector_double::size_type get_nec() const{
        return 10;
    }

    // Number of inequality constraints.
    pagmo::vector_double::size_type get_nic() const{
        return 10;
    }

    pagmo::vector_double::size_type get_nix() const
    {
        return 80;
    }

    // Implementation of the objective function and all constraints
    pagmo::vector_double fitness(const pagmo::vector_double &x_ij) const{
//        dynamic variables: x_i,j where i in I and j in K

        //constants
        int B = 30; //size of bins
        double avail_K = 8; // number of bins
        int I = 10; // number of Items
        std::vector<int> s{4,5,6,7,8,9,10,11,12,13}; //size of items

//        variables
        double min_K = 0; //how many bins are used
        std::vector<int> y_j(avail_K,0);

//        setting y corresponding to X
        for (int i = 0; i < I; ++i) {
            for (int j = 0; j < avail_K; ++j) {
                y_j[j] += x_ij[i*avail_K + j];
            }
        }

//         objective function calculation
        for(auto& y: y_j){
            min_K += y;
        }

//         make return vector and adding objective
        pagmo::vector_double return_vector{min_K};

//        adding equality constraints - every task i in I has to be scheduled once.
        for (int i = 0; i< I; ++i) {
            int temp_x_sum  = 0;
            for (int j = 0; j < avail_K; ++j) {
                temp_x_sum += x_ij[i*avail_K + j];
            }
            return_vector.emplace_back(temp_x_sum - 1);
        }

//         adding inequality constraints with respect to at least one bin
        return_vector.emplace_back(1 - min_K);
        return_vector.emplace_back(min_K - avail_K);

//         adding inequalities with respect to size of bins
        for (int j = 0; j < avail_K; ++j) {
            int temp_bin_occupation = 0;
            for (int i = 0; i < I; ++i) {
                temp_bin_occupation += s[i] * x_ij[i*avail_K + j];
            }
            return_vector.emplace_back(temp_bin_occupation - (B * y_j[j]));
        }

        return return_vector;
    }

    std::pair<pagmo::vector_double, pagmo::vector_double> get_bounds() const
    {
        std::vector<double> lower(80, 0);
        std::vector<double> upper(80, 1);
        return {lower, upper};
    }

};

struct bin_packing2 {
    // Number of equality constraints.
    pagmo::vector_double::size_type get_nec() const{
        return 0;
    }

    // Number of inequality constraints.
    pagmo::vector_double::size_type get_nic() const{
        return 10;
    }

    pagmo::vector_double::size_type get_nix() const
    {
        return 8;
    }

    // Implementation of the objective function and all constraints
    pagmo::vector_double fitness(const pagmo::vector_double &x_i) const{
//        dynamic variables: x_i,j where i in I and j in K

        //constants
        int B = 100; //size of bins
        int avail_K = 8; // number of bins
        int I = 10; // number of Items
        std::vector<int> s{4,5,6,7,8,9,10,11,12,13}; //size of items

//        variables
        double min_K = 0; //how many bins are used
        std::vector<int> y_j(avail_K,0); // if a certain bin is used


//        setting y corresponding to X
        for (int i = 0; i < I; ++i) {
            for (int j = 0; j < avail_K; ++j) {
                if (x_i[i] - j < 0.00001){
                    y_j[j] = 1;
                }
            }
        }

//         objective function calculation
        for(auto& y: y_j){
            min_K += y;
        }

//         make return vector and adding objective
        pagmo::vector_double return_vector{min_K};

//         adding inequality constraints with respect to at least one bin
        return_vector.emplace_back(1 - min_K);
        return_vector.emplace_back(min_K - avail_K);

//         adding inequalities with respect to size of bins
        for (int j = 0; j < avail_K; ++j) {
            int temp_bin_occupation = 0;
            for (int i = 0; i < I; ++i) {
                if (x_i[i] == j){
                    temp_bin_occupation += s[i] ;
                }
            }
            return_vector.emplace_back(temp_bin_occupation - B);
        }

        return return_vector;
    }

    std::pair<pagmo::vector_double, pagmo::vector_double> get_bounds() const
    {
        std::vector<double> lower(8, 0);
        std::vector<double> upper(8, 7);
        return {lower, upper};
    }

};


int main(){
    pagmo::problem prob{bin_packing2{}};
    std::cout << prob <<std::endl;

    pagmo::algorithm algo{pagmo::ihs(100)};

    // 3 - Define the topology of the archipelago
    pagmo::fully_connected topo{};

    // 4 - Instantiate an archipelago with 16 islands having each 100 individuals.
    pagmo::archipelago archi{topo,8, algo, prob, 1000};

    // 5 - Run the evolution in parallel on the 16 separate islands 10 times.
    archi.evolve(100);

    // 6 - Wait for the evolutions to finish.
    archi.wait_check();

    // 7 - Print the fitness of the best solution in each island.
    for (const auto &isl : archi) {
        std::cout << isl.get_population().champion_f()[0] << '\n';
    }
}
