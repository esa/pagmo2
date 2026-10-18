#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/nsga2.hpp>
#include <pagmo/batch_evaluators/thread_bfe.hpp>
#include <pagmo/bfe.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/threading.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

// A bi-objective travelling salesman problem, encoded with random keys
// as in the biased random-key genetic algorithms (BRKGA).
struct random_key_tsp {
    explicit random_key_tsp(std::size_t n = 30u, unsigned seed = 42u) : m_d1(n, vector_double(n)), m_d2(m_d1)
    {
        // Two sets of random points in the unit square, one per objective.
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> dist(0., 1.);
        std::vector<std::pair<double, double>> p1(n), p2(n);
        for (std::size_t i = 0; i < n; ++i) {
            p1[i] = {dist(rng), dist(rng)};
            p2[i] = {dist(rng), dist(rng)};
        }
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                m_d1[i][j] = std::hypot(p1[i].first - p1[j].first, p1[i].second - p1[j].second);
                m_d2[i][j] = std::hypot(p2[i].first - p2[j].first, p2[i].second - p2[j].second);
            }
        }
    }

    // The decoder. The random keys are sorted to obtain a tour, which is then
    // improved with a 2-opt local search accepting only the moves that improve
    // one objective without worsening the other. If rewrite is true, the keys
    // are rewritten so that they encode the improved tour.
    vector_double decode(vector_double &keys, bool rewrite) const
    {
        const auto n = keys.size();
        std::vector<std::size_t> tour(n);
        std::iota(tour.begin(), tour.end(), std::size_t(0));
        std::sort(tour.begin(), tour.end(), [&keys](std::size_t a, std::size_t b) { return keys[a] < keys[b]; });

        bool improved = true;
        while (improved) {
            improved = false;
            for (std::size_t i = 0; i + 2u < n && !improved; ++i) {
                for (std::size_t j = i + 2u; j < n && !improved; ++j) {
                    const auto a = tour[i], b = tour[i + 1u], c = tour[j], d = tour[(j + 1u) % n];
                    const auto delta1 = m_d1[a][c] + m_d1[b][d] - m_d1[a][b] - m_d1[c][d];
                    const auto delta2 = m_d2[a][c] + m_d2[b][d] - m_d2[a][b] - m_d2[c][d];
                    if (delta1 <= 1e-12 && delta2 <= 1e-12 && (delta1 < -1e-12 || delta2 < -1e-12)) {
                        std::reverse(tour.begin() + static_cast<std::ptrdiff_t>(i + 1u),
                                     tour.begin() + static_cast<std::ptrdiff_t>(j + 1u));
                        improved = true;
                    }
                }
            }
        }

        if (rewrite) {
            // Reassign the (sorted) keys following the improved tour, so that
            // decoding the rewritten keys yields the improved tour.
            vector_double sorted_keys(keys);
            std::sort(sorted_keys.begin(), sorted_keys.end());
            for (std::size_t p = 0; p < n; ++p) {
                keys[tour[p]] = sorted_keys[p];
            }
        }

        double f1 = 0., f2 = 0.;
        for (std::size_t p = 0; p < n; ++p) {
            f1 += m_d1[tour[p]][tour[(p + 1u) % n]];
            f2 += m_d2[tour[p]][tour[(p + 1u) % n]];
        }
        return {f1, f2};
    }

    // Mandatory fitness function: it must not modify its argument.
    vector_double fitness(const vector_double &keys) const
    {
        auto tmp(keys);
        return decode(tmp, false);
    }
    // Optional fitness function rewriting the decision vector in place.
    vector_double fitness_rewrite(vector_double &keys) const
    {
        return decode(keys, true);
    }

    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {vector_double(m_d1.size(), 0.), vector_double(m_d1.size(), 1.)};
    }
    vector_double::size_type get_nobj() const
    {
        return 2u;
    }
    // The decoder has no mutable state, hence it can be invoked
    // concurrently from multiple threads.
    thread_safety get_thread_safety() const
    {
        return thread_safety::constant;
    }

    std::vector<vector_double> m_d1, m_d2;
};

int main()
{
    // 1 - Instantiate a pagmo problem from the UDP.
    problem prob{random_key_tsp{}};
    std::cout << std::boolalpha << "Has fitness rewrite: " << prob.has_fitness_rewrite() << '\n';

    // 2 - Instantiate a population, decoding (and rewriting)
    // the random initial keys in parallel.
    population pop{prob, bfe{thread_bfe{}}, 24u, 42u};

    // 3 - Instantiate NSGA-II, evaluating (and rewriting) the
    // offspring in parallel.
    nsga2 uda{100u};
    uda.set_bfe(bfe{thread_bfe{}});
    algorithm algo{uda};

    // 4 - Evolve the population.
    pop = algo.evolve(pop);

    // 5 - Each stored chromosome encodes a 2-opt local optimum and
    // is consistent with its stored fitness.
    for (population::size_type i = 0; i < pop.size(); ++i) {
        auto x = pop.get_x()[i];
        const auto f = prob.fitness_rewrite(x);
        if (f != pop.get_f()[i] || x != pop.get_x()[i]) {
            std::cout << "Inconsistent individual " << i << '\n';
            return 1;
        }
    }
    std::cout << "All the " << pop.size() << " chromosomes are consistent with their fitness.\n";
    const auto best = std::min_element(pop.get_f().begin(), pop.get_f().end());
    std::cout << "Lexicographically best fitness: [" << (*best)[0] << ", " << (*best)[1] << "]\n";
}
