#ifndef PAGMO_PARETO_HPP
#define PAGMO_PARETO_HPP

/** \file pareto.hpp
 * \brief Pareto.
 *
 * This header contains utilities used to compute non-dominated fronts
 */

#include <cassert>
#include <string>
#include <tuple>

 #include "../types.hpp"
 #include "../exceptions.hpp"


namespace pagmo{

/// Pareto-dominance
/**
 * Return true if obj1 Pareto dominates obj2, false otherwise. Minimization
 * is assumed.
 *
 * Each pair of corresponding elements in obj1 and obj2 is compared: if all
 * elements in obj1 are less or equal to the corresponding element in obj2, 
 * but at least one is different, true will be returned. Otherwise, false will be returned.
 *
 * @param[in] obj1 first vector of objectives.
 * @param[in] obj2 second vector of objectives.
 *
 * @return true if obj1 is dominating obj2, false otherwise.
 */
bool pareto_dominance(const vector_double &obj1, const vector_double &obj2) 
{
    if (obj1.size() != obj2.size()) {
        pagmo_throw(std::invalid_argument,
            "Fitness of different dimensions: " + std::to_string(obj1.size()) + 
            " and " + std::to_string(obj2.size()) +
         ": cannot determine dominance");
    }
    vector_double::size_type count1 = 0u; 
    vector_double::size_type count2 = 0u;
    for (decltype(obj1.size()) i = 0u; i < obj1.size(); ++i) {
        if (obj1[i] < obj2[i]) {
            ++count1;
        }
        if (obj1[i] == obj2[i]) {
            ++count2;
        }
    }
    return ( ( (count1+count2) == obj1.size()) && (count1 > 0u) );
}

std::vector<std::vector<vector_double::size_type>> fast_non_dominated_sorting (
    const std::vector<vector_double>                    &obj_list, 
    std::vector<vector_double::size_type>               &dom_count,
    std::vector<std::vector<vector_double::size_type>>  &dom_list,
    std::vector<vector_double::size_type>               &non_dom_rank
    )
    {
        // Initialize the return values
        std::vector<std::vector<vector_double::size_type>> non_dom_fronts(1u);
        auto N = obj_list.size();
        dom_list.resize(N);
        dom_count.resize(N);
        non_dom_rank.resize(N);

        // Start the fast non dominated sort algorithm
        for (decltype(N) i = 0u; i < N; ++i) {
            dom_list[i].clear();
            dom_count[i]=0u;
            for (decltype(N) j = 0u; j < N; ++j) {
                if (i==j) continue;
                if (pareto_dominance(obj_list[i], obj_list[j])) {
                    dom_list[i].push_back(j);
                } else if (pareto_dominance(obj_list[j], obj_list[i])) {
                    ++dom_count[i];
                }
            }
            if (dom_count[i] == 0u) {
                non_dom_rank[i] = 0u;
                non_dom_fronts[0].push_back(i);
            }
        }
        // we copy dom_count as we want to output its value at this point
        auto dom_count_copy(dom_count);
        auto current_front = non_dom_fronts[0];
        std::vector<std::vector<vector_double::size_type>>::size_type front_counter(0u);
        while(current_front.size()!=0u) {
            std::vector<vector_double::size_type> next_front;
            for (decltype(current_front.size()) p = 0u; p < current_front.size(); ++p) {
                for (decltype(dom_list[current_front[p]].size()) q = 0u; q < dom_list[current_front[p]].size(); ++q) {
                    --dom_count_copy[dom_list[current_front[p]][q]];
                    if (dom_count_copy[dom_list[current_front[p]][q]] == 0u) {
                        non_dom_rank[dom_list[current_front[p]][q]] = front_counter + 1u;
                        next_front.push_back(dom_list[current_front[p]][q]);
                    }
                }
            }
            ++front_counter;
            current_front = next_front;
            if (current_front.size() != 0u) {
                non_dom_fronts.push_back(current_front);
            }
        }
        return non_dom_fronts;
    }

} // namespace pagmo
#endif