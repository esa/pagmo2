#include <iostream>
#include <utility>

#include <pagmo/problem.hpp>
#include <pagmo/types.hpp>
#include <pagmo/algorithms/gaco.hpp>
#include <pagmo/archipelago.hpp>
#include <pagmo/topologies/fully_connected.hpp>
#include <pagmo/algorithms/ihs.hpp>
#include <EGOP.h>

struct bin_packing {
    // constants of the bin packing problem (bpp)
    int size_bins; // size and number of available bins
    int num_bins;
    std::vector<int> items; // items that have to be packed
    int num_items = int(items.size());

    // Number of inequality constraints.
    pagmo::vector_double::size_type get_nic() const {
        return pagmo::vector_double::size_type(num_bins);
    }

    // Integer dimension of the problem - i.e. where each item is packed
    pagmo::vector_double::size_type get_nix() const {
        return pagmo::vector_double::size_type(num_items);
    }

    // Implementation of the objective function and all constraints
    pagmo::vector_double fitness(const pagmo::vector_double &x_i) const {
        // variables
        double num_bins_used = 0; //how many bins are used
        std::vector<int> bin_used(num_bins, 0);

        // where x_i is the bin in which an item i is placed
        // determine if a bin is used based on x_i
        for (int i = 0; i < num_items; ++i) {
            for (int j = 0; j < num_bins; ++j) {
                if ( x_i[i] == j){
                    bin_used[j] = 1;
                }
            }
        }

        // Objective function calculation; i.e. how many bins are used?
        // We want to minimize this!
        for(auto& y: bin_used){
            num_bins_used += y;
        }

        // Make return vector and adding objective
        pagmo::vector_double return_vector{num_bins_used};

        // Adding inequalities to stick to the maximum size of the bin
        for (int j = 0; j < num_bins; ++j) {
            int temp_bin_occupation = 0;
            for (int i = 0; i < num_items; ++i) {
                if (x_i[i] == j){
                    temp_bin_occupation += items[i] ; // <= 0
                }
            }
            return_vector.emplace_back(temp_bin_occupation - size_bins);
        }
        return return_vector;
    }

    // The lower and upper bounds are the bins in which an item can be placed.
    std::pair<pagmo::vector_double, pagmo::vector_double> get_bounds() const {
        std::vector<double> lower(num_items, 0);
        std::vector<double> upper(num_items, num_bins-1);
        return {lower, upper};
    }
};

int main(){

    int size_bins = 7;
    int num_bins = 6;
    std::vector<int> items{3,2,5,1,4,1,1,1,1};

    // 1 - Define the topology of the archipelago
    pagmo::problem prob{bin_packing{size_bins, num_bins, items}};

    // 2 - Define the topology of the archipelago
    pagmo::algorithm algo{pagmo::ihs(100)};

//    // 3 - Define the topology of the archipelago
//    pagmo::fully_connected topo{};
//
//    // 4 - Instantiate an archipelago with 16 islands having each 100 individuals.
//    pagmo::archipelago archi{topo, 16, algo, prob, 100};
//
//    // 5 - Run the evolution in parallel on the 8 separate islands 10 times.
//    archi.evolve(100);
//
//    // 6 - Wait for the evolutions to finish.
//    archi.wait_check();
//
//    // 7 - Get the best solution across all islands.
//    // The first number in the champion vector is the number of islands used.
//    pagmo::population champion;
//    for (const auto &isl : archi) {
//        if (champion.size() == 0){
//            champion = isl.get_population();
//        }
//        if (champion.champion_f()[0] > isl.get_population().champion_f()[0]){
//            champion = isl.get_population();
//        }
//    }
//
//    // 8 - Print the solution
//    for (int i=0; i < champion.champion_x().size(); i++){
//        std::cout << "Item " << i << " is mapped to: " << champion.champion_x()[i] << std::endl;
//    }
//    // 9 - Print fitness value
//    std::cout << "Number of Islands used: " <<   champion.champion_f()[0] << std::endl;
//
//    // 10 - Print space left in each bin.
//    for (int i=1; i < champion.champion_f().size(); i++){
//        std::cout << "Space left in bin " << i << " : " << -champion.champion_f()[i] << std::endl;
//    }
}
