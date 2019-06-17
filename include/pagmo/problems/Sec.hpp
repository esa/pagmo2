#ifndef PAGMO_PROBLEM_Sec_HPP
#define PAGMO_PROBLEM_Sec_HPP

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pagmo/detail/constants.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/types.hpp>
#define node 47266
#define cand 109
static std::vector<double> v_length, v_demand, v_fc;
static std::vector<double> v_eLen;
static std::vector<int> v_sNode, v_eNode;
static std::vector<double> upper;
static double qjv[cand][node];
namespace pagmo
{

class Sec
{

public:
    const int pmax = 30, e = 15000, uc = 10000, Nmax = 10000;
    const double capuav = 60000 * 8 * 0.8;

    static void DataConstruction()
    {
        std::string line;
        double x;
        std::ifstream myfile("data.txt");
        if (myfile.is_open()) {
            while (myfile.good()) {
                getline(myfile, line);
                std::stringstream ss(line);
                ss >> x;
                v_length.push_back(x);
                ss >> x;
                v_sNode.push_back((int)x - 1);
                ss >> x;
                v_eNode.push_back((int)x - 1);
                ss >> x;
                v_eLen.push_back(x);
                ss >> x;
                v_demand.push_back(x);
            }
            myfile.close();
        }

        std::ifstream myfile2("qjv.txt");
        if (myfile2.is_open()) {
            for (unsigned int i = 0; i < cand; i++) {
                getline(myfile2, line);
                std::stringstream ss(line);
                for (unsigned int j = 0; j < 47266; j++) {
                    ss >> x;
                    qjv[i][j] = x;
                }
            }
            myfile2.close();
        }

        std::ifstream myfile3("fc.txt");
        if (myfile3.is_open()) {
            getline(myfile3, line);
            std::stringstream ss(line);
            for (unsigned int j = 0; j < cand; j++) {
                ss >> x;
                v_fc.push_back(x);
            }
            myfile3.close();
        }

        std::ifstream myfile4("upper.txt");
        if (myfile4.is_open()) {
            getline(myfile4, line);
            std::stringstream ss(line);
            for (unsigned int j = 0; j < 109 * 2; j++) {
                ss >> x;
                upper.push_back(x);
            }
            myfile4.close();
        }
    }
    /** Constructor
     *
     * Will construct one problem from the ZDT test-suite.
     *
     * @param prob_id problem number. Must be in [1, .., 6]
     * @param param problem parameter, representing the problem dimension
     * except for ZDT5 where it represents the number of binary strings
     *
     * @throws std::invalid_argument if \p id is not in [1,..,6]
     * @throws std::invalid_argument if \p param is not at least 2.
     */

    Sec (unsigned int prob_id = 1u, unsigned int param = 2*cand) : m_prob_id(prob_id), m_param(param){};
    /// Fitness computation
    /**
     * Computes the fitness for this UDP
     *
     * @param x the decision vector.
     *
     * @return the fitness of \p x.
     */
    vector_double fitness(const vector_double &chrome) const
    {
        std::vector<double> f(2, 0);
        size_t kk;
        unsigned int i, j, flag;
        double Satisfied, Travelled, Lost;
        std::vector<int> v_closeInd(node, 0);
        std::vector<double> v_closeDis(node, 0);
        std::vector<double> v_aveDis(v_length.size() * 2, 0); // average distance for each partition ending with i
        std::vector<double> v_assDem(v_length.size() * 2, 0); // The demand for  partition ending with i
        static std::vector<int> v_sourceInd(v_length.size() * 2,
                                            0); // the facility assigned to partition ending to ith row of data
        std::vector<std::tuple<double, double, int>> v;
        // while (accumulate(chrome.begin() + cand, chrome.end(), 0) > Nmax) {
        //    for (kk = cand;kk < 2 * cand; kk++) {
        //        if (chrome[kk] > 0) {
        //            chrome[kk]--;
        //            if (chrome[kk] == 0) {
        //                chrome[kk - cand] = 0;
        //            }
        //            continue;
        //        }
        //    }
        //}
        double x, g;
        for (i = 0; i < cand; i++) {
            f[0] += v_fc[i] * chrome[i];
        }
        for (i = cand; i < 2*cand; i++) {
            f[0] += uc * chrome[i];
        }

        for (i = 0; i < node; i++) {
            flag = 0;
            for (j = 0; j < cand; j++) {
                if (chrome[j] == 1) {
                    if (flag == 0) {
                        v_closeInd[i] = j;
                        v_closeDis[i] = qjv[j][i];
                        flag = 1;
                    } else if (qjv[j][i] < v_closeDis[i]) {
                        v_closeInd[i] = j;
                        v_closeDis[i] = qjv[j][i];
                    }
                }
            }
        }

        std::vector<double> v_capacity(cand, 0);
        for (kk = 0; kk < v_capacity.size(); kk++) {
            v_capacity[kk] = chrome[v_capacity.size() + kk] * capuav * chrome[kk];
        }
        // calculation of average distance for each node in the data matrix
        for (i = 0; i < v_length.size(); i++) {
            if (v_closeInd[v_sNode[i]] == v_closeInd[v_eNode[i]]) {
                v_aveDis[i] = sqrt(
                    (pow(v_closeDis[v_sNode[i]], 2) + pow(v_closeDis[v_eNode[i]], 2) - pow(v_eLen[i], 2) / 2.0)
                    / 2.0);
                v_assDem[i] = v_demand[i];
                v_sourceInd[i] = v_closeInd[v_sNode[i]];
            } else {
                // dii'jj'
                x = (v_eLen[i]* (pow(qjv[v_closeInd[v_eNode[i]]][v_sNode[i]], 2)- pow(v_closeDis[v_sNode[i]], 2)))/ (pow(qjv[v_closeInd[v_sNode[i]]][v_eNode[i]], 2) - pow(v_closeDis[v_eNode[i]], 2)+ pow(qjv[v_closeInd[v_eNode[i]]][v_sNode[i]], 2)- pow(v_closeDis[v_sNode[i]], 2));
                // distance between breaking point and  j
                g = sqrt((x * pow(qjv[v_closeInd[v_sNode[i]]][v_eNode[i]], 2)
                          + (v_eLen[i] - x) * pow(v_closeDis[v_sNode[i]], 2)
                          - x * (v_eLen[i] - x) * v_eLen[i])
                         / v_eLen[i]);
                // average distance for this partition
                v_aveDis[i] = sqrt((pow(v_closeDis[v_sNode[i]], 2) + pow(g, 2) - pow(x, 2) / 2.0) / 2.0);
                v_assDem[i] = v_demand[i] * x / v_eLen[i];
                v_sourceInd[i] = v_closeInd[v_sNode[i]];
                v_aveDis[i + v_length.size()]
                    = sqrt((pow(v_closeDis[v_eNode[i]], 2) + pow(g, 2) - pow((v_eLen[i] - x), 2) / 2.0) / 2.0);
                v_assDem[i + v_length.size()] = v_demand[i] * (v_eLen[i] - x) / v_eLen[i];
                v_sourceInd[i + v_length.size()] = v_closeInd[v_eNode[i]];
            }
        }
        for (i = 0; i < v_aveDis.size(); i++) {
            v.push_back(std::make_tuple(v_aveDis[i], v_assDem[i], v_sourceInd[i]));
        }
        sort(v.begin(), v.end()); // sorting v based on average distance
        // assigining the capacities from small distance to highest
        Satisfied = 0;
        Travelled = 0;
        Lost = 0;
        std::vector<double> v_capacity2(v_capacity.size(), 0);
        for (i = 0; i < v_capacity.size(); i++) {
            v_capacity2[i] = v_capacity[i];
        }
        for (i = 0; i < v.size(); i++) {
            if (v_capacity2[std::get<2>(v[i])] == 0) {
                Lost += std::get<1>(v[i]);

                continue;
            }
            if (v_capacity2[std::get<2>(v[i])] >= std::get<0>(v[i]) * std::get<1>(v[i])) {
                v_capacity2[std::get<2>(v[i])] -= std::get<0>(v[i]) * std::get<1>(v[i]);
                Satisfied += std::get<1>(v[i]);
                Travelled += std::get<0>(v[i]) * std::get<1>(v[i]);
            } else {
                Satisfied = Satisfied + v_capacity2[std::get<2>(v[i])] / std::get<0>(v[i]);
                Lost = Lost + std::get<1>(v[i]) - (v_capacity2[std::get<2>(v[i])] / std::get<0>(v[i]));
                Travelled += v_capacity2[std::get<2>(v[i])];
                v_capacity2[std::get<2>(v[i])] = 0;
            }
        }
        double demand = 0, dis = 0;
        for (i = 0; i < v.size(); i++) {
            demand += std::get<1>(v[i]);
            dis += std::get<0>(v[i]);
        }

        f[1] = Lost;
        return f;
    }
    /// Number of objectives
    /**
     *
     * It returns the number of objectives.
     *
     * @return the number of objectives
     */
    vector_double::size_type get_nobj() const
    {
        return 2u;
    }

    /// Box-bounds
    /**
     *
     * It returns the box-bounds for this UDP.
     *
     * @return the lower and upper bounds for each of the decision vector components
     */
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {vector_double(m_param, 0.), vector_double(m_param, 1.)};
    }

    /// Integer dimension
    /**
     * It returns the integer dimension for this UDP.
     *
     * @return the integer dimension of the UDP
     */
    vector_double::size_type get_nix() const
    {
        return 2 * cand;
    }

    /// Problem name
    /**
     * @return a string containing the problem name
     */
    std::string get_name() const
    {
        return "Sec" + std::to_string(m_prob_id);
    }
    /// Distance from the Pareto front (of a population)
    /**
     * Convergence metric for a given population (0 = on the optimal front)
     *
     * Takes the average across the input population of the p_distance
     *
     * @param pop population to be assigned a pareto distance
     * @return the p_distance
     *
     */

    template <typename Archive>
    void serialize(Archive &ar)
    {
        ar(m_prob_id, m_param);
    }

private:
    // Problem dimensions

    unsigned int m_prob_id;
    unsigned int m_param;
};
} // namespace pagmo

PAGMO_REGISTER_PROBLEM(pagmo::Sec)

#endif
