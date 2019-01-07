/* Copyright 2017-2018 PaGMO development team

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

#ifndef PAGMO_MULTI_OBJECTIVE_HPP
#define PAGMO_MULTI_OBJECTIVE_HPP

/** \file multi_objective.hpp
 * \brief Multi objective optimization utilities.
 *
 * This header contains utilities used to compute non dominated fronts and other
 * quantities useful for multi objective optimization
 */

#include <algorithm>
#include <boost/numeric/conversion/cast.hpp>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <pagmo/detail/custom_comparisons.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/discrepancy.hpp> // halton

namespace pagmo
{

namespace detail
{
// Recursive function building all m-ple of elements of X summing to s
// In C/C++ implementations there exists a limit on the number of times you
// can call recursively a function. It depends on a variety of factors,
// but probably it a number around few thousands on modern machines.
// If the limit is surpassed, the program terminates.
// I was thinking that one could create a problem with a few thousands objectives,
// call this function thus causing a crash from Python. In principle I think we
// can prevent this by limiting the recursion (e.g., via a function parameter that
// gets increased each time the function is called from itself).
// But for now I'd just put a note about this.
inline void reksum(std::vector<std::vector<double>> &retval, const std::vector<population::size_type> &X,
                   population::size_type m, population::size_type s, std::vector<double> eggs = std::vector<double>())
{
    if (m == 1u) {
        if (std::find(X.begin(), X.end(), s) == X.end()) { // not found
            return;
        } else {
            eggs.push_back(static_cast<double>(s));
            retval.push_back(eggs);
        }
    } else {
        for (decltype(X.size()) i = 0u; i < X.size(); ++i) {
            eggs.push_back(static_cast<double>(X[i]));
            reksum(retval, X, m - 1u, s - X[i], eggs);
            eggs.pop_back();
        }
    }
}
} // namespace detail

/// Pareto-dominance
/**
 * Return true if \p obj1 Pareto dominates \p obj2, false otherwise. Minimization
 * is assumed.
 *
 * Each pair of corresponding elements in \p obj1 and \p obj2 is compared: if all
 * elements in \p obj1 are less or equal to the corresponding element in \p obj2,
 * but at least one is different, \p true will be returned. Otherwise, \p false will be returned.
 *
 * @param obj1 first vector of objectives.
 * @param obj2 second vector of objectives.
 *
 * @return \p true if \p obj1 is dominating \p obj2, \p false otherwise.
 *
 * @throws std::invalid_argument if the dimensions of the two objectives are different
 */
inline bool pareto_dominance(const vector_double &obj1, const vector_double &obj2)
{
    if (obj1.size() != obj2.size()) {
        pagmo_throw(std::invalid_argument,
                    "Different number of objectives found in input fitnesses: " + std::to_string(obj1.size()) + " and "
                        + std::to_string(obj2.size()) + ". I cannot define dominance");
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
    return (((count1 + count2) == obj1.size()) && (count1 > 0u));
}

/// Non dominated front 2D (Kung's algorithm)
/**
 * Finds the non dominated front of a set of two dimensional objectives. Complexity is O(N logN) and is thus lower than
 * the
 * complexity of calling pagmo::fast_non_dominated_sorting
 *
 * See: Jensen, Mikkel T. "Reducing the run-time complexity of multiobjective EAs: The NSGA-II and other algorithms."
 * IEEE Transactions on Evolutionary Computation 7.5 (2003): 503-515.
 *
 * @param input_objs an <tt>std::vector</tt> containing the points (i.e. vector of objectives)
 *
 * @return A <tt>std::vector</tt> containing the indexes of the points in the non-dominated front
 *
 * @throws std::invalid_argument If the objective vectors are not all containing two-objectives
 */
inline std::vector<vector_double::size_type> non_dominated_front_2d(const std::vector<vector_double> &input_objs)
{
    // If the input is empty return an empty vector
    if (input_objs.size() == 0u) {
        return {};
    }
    // How many objectives? M, of course.
    auto M = input_objs[0].size();
    // We make sure all input_objs contain M objectives
    if (!std::all_of(input_objs.begin(), input_objs.end(),
                     [M](const vector_double &item) { return item.size() == M; })) {
        pagmo_throw(std::invalid_argument, "Input contains vector of objectives with heterogeneous dimensionalities");
    }
    // We make sure this function is only requested for two objectives.
    if (M != 2u) {
        pagmo_throw(std::invalid_argument, "The number of objectives detected is " + std::to_string(M)
                                               + ", while Kung's algorithm only works for two objectives.");
    }
    // Sanity checks are over. We may run Kung's algorithm.
    std::vector<vector_double::size_type> front;
    std::vector<vector_double::size_type> indexes(input_objs.size());
    std::iota(indexes.begin(), indexes.end(), vector_double::size_type(0u));
    // Sort in ascending order with respect to the first component
    std::sort(indexes.begin(), indexes.end(),
              [&input_objs](vector_double::size_type idx1, vector_double::size_type idx2) {
                  if (input_objs[idx1][0] == input_objs[idx2][0]) {
                      return detail::less_than_f(input_objs[idx1][1], input_objs[idx2][1]);
                  }
                  return detail::less_than_f(input_objs[idx1][0], input_objs[idx2][0]);
              });
    for (auto i : indexes) {
        bool flag = false;
        for (auto j : front) {
            if (pareto_dominance(input_objs[j], input_objs[i])) {
                flag = true;
                break;
            }
        }
        if (!flag) {
            front.push_back(i);
        }
    }
    return front;
}

/// Return type for the fast_non_dominated_sorting algorithm
using fnds_return_type
    = std::tuple<std::vector<std::vector<vector_double::size_type>>, std::vector<std::vector<vector_double::size_type>>,
                 std::vector<vector_double::size_type>, std::vector<vector_double::size_type>>;

/// Fast non dominated sorting
/**
 * An implementation of the fast non dominated sorting algorithm. Complexity is \f$ O(MN^2)\f$ where \f$M\f$ is the
 * number of objectives
 * and \f$N\f$ is the number of individuals.
 *
 * See: Deb, Kalyanmoy, et al. "A fast elitist non-dominated sorting genetic algorithm
 * for multi-objective optimization: NSGA-II." Parallel problem solving from nature PPSN VI. Springer Berlin Heidelberg,
 * 2000.
 *
 * @param points An std::vector containing the objectives of different individuals. Example
 * {{1,2,3},{-2,3,7},{-1,-2,-3},{0,0,0}}
 *
 * @return an std::tuple containing:
 *  - the non dominated fronts, an <tt>std::vector<std::vector<vector_double::size_type>></tt>
 * containing the non dominated fronts. Example {{1,2},{3},{0}}
 *  - the domination list, an <tt>std::vector<std::vector<vector_double::size_type>></tt>
 * containing the domination list, i.e. the indexes of all individuals
 * dominated by the individual at position \f$i\f$. Example {{},{},{0,3},{0}}
 *  - the domination count, an <tt>std::vector<vector_double::size_type></tt> containing the number of individuals
 * that dominate the individual at position \f$i\f$. Example {2, 0, 0, 1}
 *  - the non domination rank, an <tt>std::vector<vector_double::size_type></tt> containing the index of the non
 * dominated front to which the individual at position \f$i\f$ belongs. Example {2,0,0,1}
 *
 * @throws std::invalid_argument If the size of \p points is not at least 2
 */
inline fnds_return_type fast_non_dominated_sorting(const std::vector<vector_double> &points)
{
    auto N = points.size();
    // We make sure to have two points at least (one could also be allowed)
    if (N < 2u) {
        pagmo_throw(std::invalid_argument, "At least two points are needed for fast_non_dominated_sorting: "
                                               + std::to_string(N) + " detected.");
    }
    // Initialize the return values
    std::vector<std::vector<vector_double::size_type>> non_dom_fronts(1u);
    std::vector<std::vector<vector_double::size_type>> dom_list(N);
    std::vector<vector_double::size_type> dom_count(N);
    std::vector<vector_double::size_type> non_dom_rank(N);

    // Start the fast non dominated sort algorithm
    for (decltype(N) i = 0u; i < N; ++i) {
        dom_list[i].clear();
        dom_count[i] = 0u;
        for (decltype(N) j = 0u; j < N; ++j) {
            if (i == j) {
                continue;
            }
            if (pareto_dominance(points[i], points[j])) {
                dom_list[i].push_back(j);
            } else if (pareto_dominance(points[j], points[i])) {
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
    while (current_front.size() != 0u) {
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
    return std::make_tuple(std::move(non_dom_fronts), std::move(dom_list), std::move(dom_count),
                           std::move(non_dom_rank));
}

/// Crowding distance
/**
 * An implementation of the crowding distance. Complexity is \f$ O(MNlog(N))\f$ where \f$M\f$ is the number of
 * objectives
 * and \f$N\f$ is the number of individuals. The function assumes the input is a non-dominated front. Failiure to this
 * condition
 * will result in undefined behaviour.
 *
 * See: Deb, Kalyanmoy, et al. "A fast elitist non-dominated sorting genetic algorithm
 * for multi-objective optimization: NSGA-II." Parallel problem solving from nature PPSN VI. Springer Berlin Heidelberg,
 * 2000.
 *
 * @param non_dom_front An <tt>std::vector<vector_double></tt> containing a non dominated front. Example
 * {{0,0},{-1,1},{2,-2}}
 *
 * @returns a vector_double containing the crowding distances. Example: {2, inf, inf}
 *
 * @throws std::invalid_argument If \p non_dom_front does not contain at least two points
 * @throws std::invalid_argument If points in \p do not all have at least two objectives
 * @throws std::invalid_argument If points in \p non_dom_front do not all have the same dimensionality
 */
inline vector_double crowding_distance(const std::vector<vector_double> &non_dom_front)
{
    auto N = non_dom_front.size();
    // We make sure to have two points at least
    if (N < 2u) {
        pagmo_throw(std::invalid_argument,
                    "A non dominated front must contain at least two points: " + std::to_string(N) + " detected.");
    }
    auto M = non_dom_front[0].size();
    // We make sure the first point of the input non dominated front contains at least two objectives
    if (M < 2u) {
        pagmo_throw(std::invalid_argument, "Points in the non dominated front must contain at least two objectives: "
                                               + std::to_string(M) + " detected.");
    }
    // We make sure all points contain the same number of objectives
    if (!std::all_of(non_dom_front.begin(), non_dom_front.end(),
                     [M](const vector_double &item) { return item.size() == M; })) {
        pagmo_throw(std::invalid_argument, "A non dominated front must contain points of uniform dimensionality. Some "
                                           "different sizes were instead detected.");
    }
    std::vector<vector_double::size_type> indexes(N);
    std::iota(indexes.begin(), indexes.end(), vector_double::size_type(0u));
    vector_double retval(N, 0.);
    for (decltype(M) i = 0u; i < M; ++i) {
        std::sort(indexes.begin(), indexes.end(),
                  [i, &non_dom_front](vector_double::size_type idx1, vector_double::size_type idx2) {
                      return detail::less_than_f(non_dom_front[idx1][i], non_dom_front[idx2][i]);
                  });
        retval[indexes[0]] = std::numeric_limits<double>::infinity();
        retval[indexes[N - 1u]] = std::numeric_limits<double>::infinity();
        double df = non_dom_front[indexes[N - 1u]][i] - non_dom_front[indexes[0]][i];
        for (decltype(N - 2u) j = 1u; j < N - 1u; ++j) {
            retval[indexes[j]] += (non_dom_front[indexes[j + 1u]][i] - non_dom_front[indexes[j - 1u]][i]) / df;
        }
    }
    return retval;
}

/// Sorts a population in multi-objective optimization
/**
 * Sorts a population (intended here as an <tt>std::vector<vector_double></tt> containing the  objective vectors)
 * with respect to the following strict ordering:
 * - \f$f_1 \prec f_2\f$ if the non domination ranks are such that \f$i_1 < i_2\f$. In case
 * \f$i_1 = i_2\f$, then \f$f_1 \prec f_2\f$ if the crowding distances are such that \f$d_1 > d_2\f$.
 *
 * Complexity is \f$ O(MN^2)\f$ where \f$M\f$ is the number of objectives and \f$N\f$ is the number of individuals.
 *
 * This function will also work for single objective optimization, i.e. with 1 objective
 * in which case, though, it is more efficient to sort using directly one of the following forms:
 *
 * @code{.unparsed}
 * std::sort(input_f.begin(), input_f.end(), [] (auto a, auto b) {return a[0] < b[0];});
 * @endcode
 * @code{.unparsed}
 * std::vector<vector_double::size_type> idx(input_f.size());
 * std::iota(idx.begin(), idx.end(), vector_double::size_type(0u));
 * std::sort(idx.begin(), idx.end(), [] (auto a, auto b) {return input_f[a][0] < input_f[b][0];});
 * @endcode
 *
 * @param input_f Input objectives vectors. Example {{0.25,0.25},{-1,1},{2,-2}};
 *
 * @returns an <tt>std::vector</tt> containing the indexes of the sorted objectives vectors. Example {1,2,0}
 *
 * @throws unspecified all exceptions thrown by pagmo::fast_non_dominated_sorting and pagmo::crowding_distance
 */
inline std::vector<vector_double::size_type> sort_population_mo(const std::vector<vector_double> &input_f)
{
    if (input_f.size() < 2u) { // corner cases
        if (input_f.size() == 0u) {
            return {};
        }
        if (input_f.size() == 1u) {
            return {0u};
        }
    }
    // Create the indexes 0....N-1
    std::vector<vector_double::size_type> retval(input_f.size());
    std::iota(retval.begin(), retval.end(), vector_double::size_type(0u));
    // Run fast-non-dominated sorting and compute the crowding distance for all input objectives vectors
    auto tuple = fast_non_dominated_sorting(input_f);
    vector_double crowding(input_f.size());
    for (const auto &front : std::get<0>(tuple)) {
        if (front.size() == 1u) {
            crowding[front[0]] = 0u; // corner case of a non dominated front containing one individual. Crowding
                                     // distance is not defined nor it will be used
        } else {
            std::vector<vector_double> non_dom_fits(front.size());
            for (decltype(front.size()) i = 0u; i < front.size(); ++i) {
                non_dom_fits[i] = input_f[front[i]];
            }
            vector_double tmp(crowding_distance(non_dom_fits));
            for (decltype(front.size()) i = 0u; i < front.size(); ++i) {
                crowding[front[i]] = tmp[i];
            }
        }
    }
    // Sort the indexes
    std::sort(retval.begin(), retval.end(),
              [&tuple, &crowding](vector_double::size_type idx1, vector_double::size_type idx2) {
                  if (std::get<3>(tuple)[idx1] == std::get<3>(tuple)[idx2]) {        // same non domination rank
                      return detail::greater_than_f(crowding[idx1], crowding[idx2]); // crowding distance decides
                  } else {                                                           // different non domination ranks
                      return std::get<3>(tuple)[idx1] < std::get<3>(tuple)[idx2];    // non domination rank decides
                  };
              });
    return retval;
}

/// Selects the best N individuals in multi-objective optimization
/**
 * Selects the best N individuals out of a population, (intended here as an
 * <tt>std::vector<vector_double></tt> containing the  objective vectors). The strict ordering used
 * is the same as that defined in pagmo::sort_population_mo.
 *
 * Complexity is \f$ O(MN^2)\f$ where \f$M\f$ is the number of objectives and \f$N\f$ is the number of individuals.
 *
 * While the complexity is the same as that of pagmo::sort_population_mo, this function returns a permutation
 * of:
 *
 * @code{.unparsed}
 * auto ret = pagmo::sort_population_mo(input_f).resize(N);
 * @endcode
 *
 * but it is faster than the above code: it avoids to compute the crowidng distance for all individuals and only
 * computes
 * it for the last non-dominated front that contains individuals included in the best N.
 *
 * @param input_f Input objectives vectors. Example {{0.25,0.25},{-1,1},{2,-2}};
 * @param N Number of best individuals to return
 *
 * @returns an <tt>std::vector</tt> containing the indexes of the best N objective vectors. Example {2,1}
 *
 * @throws unspecified all exceptions thrown by pagmo::fast_non_dominated_sorting and pagmo::crowding_distance
 */
inline std::vector<vector_double::size_type> select_best_N_mo(const std::vector<vector_double> &input_f,
                                                              vector_double::size_type N)
{
    if (N < 1u) {
        pagmo_throw(std::invalid_argument,
                    "The best: " + std::to_string(N) + " individuals were requested, while 1 is the minimum");
    }
    if (input_f.size() == 0u) { // corner case
        return {};
    }
    if (input_f.size() == 1u) { // corner case
        return {0u};
    }
    if (N >= input_f.size()) { // corner case
        std::vector<vector_double::size_type> retval(input_f.size());
        std::iota(retval.begin(), retval.end(), vector_double::size_type(0u));
        return retval;
    }
    std::vector<vector_double::size_type> retval;
    std::vector<vector_double::size_type>::size_type front_id(0u);
    // Run fast-non-dominated sorting
    auto tuple = fast_non_dominated_sorting(input_f);
    // Insert all non dominated fronts if not more than N
    for (const auto &front : std::get<0>(tuple)) {
        if (retval.size() + front.size() <= N) {
            for (auto i : front) {
                retval.push_back(i);
            }
            if (retval.size() == N) {
                return retval;
            }
            ++front_id;
        } else {
            break;
        }
    }
    auto front = std::get<0>(tuple)[front_id];
    std::vector<vector_double> non_dom_fits(front.size());
    // Run crowding distance for the front
    for (decltype(front.size()) i = 0u; i < front.size(); ++i) {
        non_dom_fits[i] = input_f[front[i]];
    }
    vector_double cds(crowding_distance(non_dom_fits));
    // We now have front and crowding distance, we sort the front w.r.t. the crowding
    std::vector<vector_double::size_type> idxs(front.size());
    std::iota(idxs.begin(), idxs.end(), vector_double::size_type(0u));
    std::sort(idxs.begin(), idxs.end(), [&cds](vector_double::size_type idx1, vector_double::size_type idx2) {
        return detail::greater_than_f(cds[idx1], cds[idx2]);
    }); // Descending order1
    auto remaining = N - retval.size();
    for (decltype(remaining) i = 0u; i < remaining; ++i) {
        retval.push_back(front[idxs[i]]);
    }
    return retval;
}

/// Ideal point
/**
 * Computes the ideal point of an input population, (intended here as an
 * <tt>std::vector<vector_double></tt> containing the  objective vectors).
 *
 * Complexity is \f$ O(MN)\f$ where \f$M\f$ is the number of objectives and \f$N\f$ is the number of individuals.
 *
 * @param points Input objectives vectors. Example {{-1,3,597},{1,2,3645},{2,9,789},{0,0,231},{6,-2,4576}};
 *
 * @returns A vector_double containing the ideal point. Example: {-1,-2,231}
 *
 * @throws std::invalid_argument if the input objective vectors are not all of the same size
 */
inline vector_double ideal(const std::vector<vector_double> &points)
{
    // Corner case
    if (points.size() == 0u) {
        return {};
    }

    // Sanity checks
    auto M = points[0].size();
    for (const auto &f : points) {
        if (f.size() != M) {
            pagmo_throw(std::invalid_argument,
                        "Input vector of objectives must contain fitness vector of equal dimension "
                            + std::to_string(M));
        }
    }
    // Actual algorithm
    vector_double retval(M);
    for (decltype(M) i = 0u; i < M; ++i) {
        retval[i]
            = (*std::min_element(points.begin(), points.end(),
                                 [i](const vector_double &f1, const vector_double &f2) { return f1[i] < f2[i]; }))[i];
    }
    return retval;
}

/// Nadir point
/**
 * Computes the nadir point of an input population, (intended here as an
 * <tt>std::vector<vector_double></tt> containing the  objective vectors).
 *
 * Complexity is \f$ O(MN^2)\f$ where \f$M\f$ is the number of objectives and \f$N\f$ is the number of individuals.
 *
 * @param points Input objective vectors. Example {{0,7},{1,5},{2,3},{4,2},{7,1},{10,0},{6,6},{9,15}}
 *
 * @returns A vector_double containing the nadir point. Example: {10,7}
 *
 */
inline vector_double nadir(const std::vector<vector_double> &points)
{
    // Corner case
    if (points.size() == 0u) {
        return {};
    }
    // Sanity checks
    auto M = points[0].size();
    // We extract all objective vectors belonging to the first non dominated front (the Pareto front)
    auto pareto_idx = std::get<0>(fast_non_dominated_sorting(points))[0];
    std::vector<vector_double> nd_points;
    for (auto idx : pareto_idx) {
        nd_points.push_back(points[idx]);
    }
    // And compute the nadir over them
    vector_double retval(M);
    for (decltype(M) i = 0u; i < M; ++i) {
        retval[i]
            = (*std::max_element(nd_points.begin(), nd_points.end(),
                                 [i](const vector_double &f1, const vector_double &f2) { return f1[i] < f2[i]; }))[i];
    }
    return retval;
}

/// Decomposition weights generation
/**
 * Generates a requested number of weight vectors to be used to decompose a multi-objective problem. Three methods are
 *available:
 * - "grid" generates weights on an uniform grid. This method may only be used when the number of requested weights to
 *be genrated is such that a uniform grid is indeed possible. In
 * two dimensions this is always the case, but in larger dimensions uniform grids are possible only in special cases
 * - "random" generates weights randomly distributing them uniformly on the simplex (weights are such that \f$\sum_i
 * \lambda_i = 1\f$)
 * - "low discrepancy" generates weights using a low-discrepancy sequence to, eventually, obtain a
 * better coverage of the Pareto front. Halton sequence is used since low dimensionalities are expected in the number of
 * objectives (i.e. less than 20), hence Halton sequence is deemed as appropriate.
 *
 * \verbatim embed:rst:leading-asterisk
 * .. note::
 *
 *    All genration methods are guaranteed to generate weights on the simplex (:math:`\sum_i \lambda_i = 1`). All
 *    weight generation methods are guaranteed to generate the canonical weights [1,0,0,...], [0,1,0,..], ... first.
 *
 * \endverbatim
 *
 * Example: to generate 10 weights distributed somehow regularly to decompose a three dimensional problem:
 * @code{.unparsed}
 * detail::random_engine_type r_engine();
 * auto lambdas = decomposition_weights(3u, 10u, "low discrepancy", r_engine);
 * @endcode
 *
 * @param n_f dimension of each weight vector (i.e. fitness dimension)
 * @param n_w number of weights to be generated
 * @param method methods to generate the weights of the decomposed problems. One of "grid", "random",
 *"low discrepancy"
 * @param r_engine random engine
 *
 * @returns an <tt>std:vector</tt> containing the weight vectors
 *
 * @throws if \p nf and \p nw are not compatible with the selected weight generation method or if \p method
 * is not one of "grid", "random" or "low discrepancy"
 */
inline std::vector<vector_double> decomposition_weights(vector_double::size_type n_f, vector_double::size_type n_w,
                                                        const std::string &method, detail::random_engine_type &r_engine)
{
    // Sanity check
    if (n_f > n_w) {
        pagmo_throw(std::invalid_argument,
                    "A fitness size of " + std::to_string(n_f)
                        + " was requested to the weight generation routine, while " + std::to_string(n_w)
                        + " weights were requested to be generated. To allow weight be generated correctly the number "
                          "of weights must be strictly larger than the number of objectives");
    }

    if (n_f < 2u) {
        pagmo_throw(
            std::invalid_argument,
            "A fitness size of " + std::to_string(n_f)
                + " was requested to generate decomposed weights. A dimension of at least two must be requested.");
    }

    // Random distributions
    std::uniform_real_distribution<double> drng(0., 1.); // to generate a number in [0, 1)
    std::vector<vector_double> retval;
    if (method == "grid") {
        // find the largest H resulting in a population smaller or equal to NP
        decltype(n_w) H;
        if (n_f == 2u) {
            H = n_w - 1u;
        } else if (n_f == 3u) {
            H = static_cast<decltype(H)>(std::floor(0.5 * (std::sqrt(8. * static_cast<double>(n_w) + 1.) - 3.)));
        } else {
            H = 1u;
            while (binomial_coefficient(H + n_f - 1u, n_f - 1u) <= static_cast<double>(n_w)) {
                ++H;
            }
            H--;
        }
        // We check that NP equals the population size resulting from H
        if (std::abs(static_cast<double>(n_w) - binomial_coefficient(H + n_f - 1u, n_f - 1u)) > 1E-8) {
            std::ostringstream error_message;
            error_message << "Population size of " << std::to_string(n_w) << " is detected, but not supported by the '"
                          << method << "' weight generation method selected. A size of "
                          << binomial_coefficient(H + n_f - 1u, n_f - 1u) << " or "
                          << binomial_coefficient(H + n_f, n_f - 1u) << " is possible.";
            pagmo_throw(std::invalid_argument, error_message.str());
        }
        // We generate the weights
        std::vector<population::size_type> range(H + 1u);
        std::iota(range.begin(), range.end(), std::vector<population::size_type>::size_type(0u));
        detail::reksum(retval, range, n_f, H);
        for (decltype(retval.size()) i = 0u; i < retval.size(); ++i) {
            for (decltype(retval[i].size()) j = 0u; j < retval[i].size(); ++j) {
                retval[i][j] /= static_cast<double>(H);
            }
        }
    } else if (method == "low discrepancy") {
        // We first push back the "corners" [1,0,0,...], [0,1,0,...]
        for (decltype(n_f) i = 0u; i < n_f; ++i) {
            retval.push_back(vector_double(n_f, 0.));
            retval[i][i] = 1.;
        }
        // Then we add points on the simplex randomly genrated using Halton low discrepancy sequence
        halton ld_seq{boost::numeric_cast<unsigned int>(n_f - 1u), boost::numeric_cast<unsigned int>(n_f)};
        for (decltype(n_w) i = n_f; i < n_w; ++i) {
            retval.push_back(sample_from_simplex(ld_seq()));
        }
    } else if (method == "random") {
        // We first push back the "corners" [1,0,0,...], [0,1,0,...]
        for (decltype(n_f) i = 0u; i < n_f; ++i) {
            retval.push_back(vector_double(n_f, 0.));
            retval[i][i] = 1.;
        }
        for (decltype(n_w) i = n_f; i < n_w; ++i) {
            vector_double dummy(n_f - 1u, 0.);
            for (decltype(n_f) j = 0u; j < n_f - 1u; ++j) {
                dummy[j] = drng(r_engine);
            }
            retval.push_back(sample_from_simplex(dummy));
        }
    } else {
        pagmo_throw(std::invalid_argument,
                    "Weight generation method " + method
                        + " is unknown. One of 'grid', 'random' or 'low discrepancy' was expected");
    }
    return retval;
}

/// Decomposes a vector of objectives.
/**
 * A vector of objectives is reduced to one only objective using a decomposition
 * technique.
 *
 * Three different *decomposition methods* are here made available:
 *
 * - weighted decomposition,
 * - Tchebycheff decomposition,
 * - boundary interception method (with penalty constraint).
 *
 * In the case of \f$n\f$ objectives, we indicate with: \f$ \mathbf f(\mathbf x) = [f_1(\mathbf x), \ldots,
 * f_n(\mathbf x)] \f$ the vector containing the original multiple objectives, with: \f$ \boldsymbol \lambda =
 * (\lambda_1, \ldots, \lambda_n) \f$ an \f$n\f$-dimensional weight vector and with: \f$ \mathbf z^* = (z^*_1, \ldots,
 * z^*_n) \f$ an \f$n\f$-dimensional reference point. We also ussume \f$\lambda_i > 0, \forall i=1..n\f$ and \f$\sum_i
 * \lambda_i = 1\f$.
 *
 * The resulting single objective is thus defined as:
 *
 * - weighted decomposition: \f$ f_d(\mathbf x) = \boldsymbol \lambda \cdot \mathbf f \f$,
 * - Tchebycheff decomposition: \f$ f_d(\mathbf x) = \max_{1 \leq i \leq m} \lambda_i \vert f_i(\mathbf x) - z^*_i \vert
 * \f$,
 * - boundary interception method (with penalty constraint): \f$ f_d(\mathbf x) = d_1 + \theta d_2\f$,
 *
 * where \f$d_1 = (\mathbf f - \mathbf z^*) \cdot \hat {\mathbf i}_{\lambda}\f$,
 * \f$d_2 = \vert (\mathbf f - \mathbf z^*) - d_1 \hat {\mathbf i}_{\lambda})\vert\f$ and
 * \f$ \hat {\mathbf i}_{\lambda} = \frac{\boldsymbol \lambda}{\vert \boldsymbol \lambda \vert}\f$.
 *
 * @param f input vector of objectives.
 * @param weight the weight to be used in the decomposition.
 * @param ref_point the reference point to be used if either "tchebycheff" or "bi".
 * was indicated as a decomposition method. Its value is ignored if "weighted" was indicated.
 * @param method decomposition method: one of "weighted", "tchebycheff" or "bi"
 *
 * @return the decomposed objective.
 *
 * @throws std::invalid_argument if \p f, \p weight and \p ref_point have different sizes
 * @throws std::invalid_argument if \p method is not one of "weighted", "tchebycheff" or "bi"
 */
inline vector_double decompose_objectives(const vector_double &f, const vector_double &weight,
                                          const vector_double &ref_point, const std::string &method)
{
    if (weight.size() != f.size()) {
        pagmo_throw(std::invalid_argument,
                    "Weight vector size must be equal to the number of objectives. The size of the weight vector is "
                        + std::to_string(weight.size()) + " while " + std::to_string(f.size())
                        + " objectives were detected");
    }
    if (ref_point.size() != f.size()) {
        pagmo_throw(
            std::invalid_argument,
            "Reference point size must be equal to the number of objectives. The size of the reference point is "
                + std::to_string(ref_point.size()) + " while " + std::to_string(f.size())
                + " objectives were detected");
    }
    if (f.size() == 0u) {
        pagmo_throw(std::invalid_argument, "The number of objectives detected is: " + std::to_string(f.size())
                                               + ". Cannot decompose this into anything.");
    }
    double fd = 0.;
    if (method == "weighted") {
        for (decltype(f.size()) i = 0u; i < f.size(); ++i) {
            fd += weight[i] * f[i];
        }
    } else if (method == "tchebycheff") {
        double tmp, fixed_weight;
        for (decltype(f.size()) i = 0u; i < f.size(); ++i) {
            (weight[i] == 0.) ? (fixed_weight = 1e-4)
                              : (fixed_weight = weight[i]); // fixes the numerical problem of 0 weights
            tmp = fixed_weight * std::abs(f[i] - ref_point[i]);
            if (tmp > fd) {
                fd = tmp;
            }
        }
    } else if (method == "bi") { // BI method
        const double THETA = 5.;
        double d1 = 0.;
        double weight_norm = 0.;
        for (decltype(f.size()) i = 0u; i < f.size(); ++i) {
            d1 += (f[i] - ref_point[i]) * weight[i];
            weight_norm += std::pow(weight[i], 2);
        }
        weight_norm = std::sqrt(weight_norm);
        d1 = d1 / weight_norm;

        double d2 = 0.;
        for (decltype(f.size()) i = 0u; i < f.size(); ++i) {
            d2 += std::pow(f[i] - (ref_point[i] + d1 * weight[i] / weight_norm), 2);
        }
        d2 = std::sqrt(d2);
        fd = d1 + THETA * d2;
    } else {
        pagmo_throw(std::invalid_argument, "The decomposition method chosen was: " + method
                                               + R"(, but only "weighted", "tchebycheff" or "bi" are allowed)");
    }
    return {fd};
}

} // namespace pagmo
#endif
