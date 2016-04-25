#ifndef PAGMO_PARETO_HPP
#define PAGMO_PARETO_HPP

/** \file pareto.hpp
 * \brief Pareto.
 *
 * This header contains utilities used to compute non-dominated fronts
 */

#include <limits>
#include <string>
#include <tuple>

 #include "../types.hpp"
 #include "../exceptions.hpp"
  #include "../io.hpp"


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
         ": cannot define dominance");
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

std::tuple<
    std::vector<std::vector<vector_double::size_type>>,
    std::vector<std::vector<vector_double::size_type>>,
    std::vector<vector_double::size_type>,
    std::vector<vector_double::size_type>
> fast_non_dominated_sorting (const std::vector<vector_double> &obj_list)
    {
        // Initialize the return values
        auto N = obj_list.size();
        std::vector<std::vector<vector_double::size_type>> non_dom_fronts(1u);
        std::vector<std::vector<vector_double::size_type>> dom_list(N);
        std::vector<vector_double::size_type> dom_count(N);
        std::vector<vector_double::size_type> non_dom_rank(N);

        // Start the fast non dominated sort algorithm
        for (decltype(N) i = 0u; i < N; ++i) {
            dom_list[i].clear();
            dom_count[i]=0u;
            for (decltype(N) j = 0u; j < N; ++j) {
                if (i==j) {
                    continue;
                }
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
        return std::make_tuple(std::move(non_dom_fronts), std::move(dom_list), std::move(dom_count), std::move(non_dom_rank));
    }

vector_double crowding_distance(const std::vector<vector_double> &non_dom_front)
{
    auto N = non_dom_front.size();
    auto M = non_dom_front[0].size();
    std::vector<vector_double::size_type> indexes(N);
    std::iota(indexes.begin(), indexes.end(), 0);
    // TODO check size is not zero, check M are all the same
    vector_double retval(N,0);
    for (decltype(M) i=0u; i < M; ++i) {
        std::sort(indexes.begin(), indexes.end(), [i, non_dom_front] (vector_double::size_type idx1, vector_double::size_type idx2) {return non_dom_front[idx1][i] < non_dom_front[idx2][i];});
        retval[indexes[0]] = std::numeric_limits<double>::infinity();
        retval[indexes[N-1]] =  std::numeric_limits<double>::infinity();
        double df = non_dom_front[indexes[N-1]][i] - non_dom_front[indexes[0]][i];
        for (decltype(N-2u) j=1u; j < N-1u; ++j) {
            retval[indexes[j]] += (non_dom_front[indexes[j+1]][i] - non_dom_front[indexes[j-1]][i]) / df;
        }
    }
    return retval;
}

} // namespace pagmo
#endif