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

#ifndef PAGMO_PROBLEM_CEC2013_HPP
#define PAGMO_PROBLEM_CEC2013_HPP

#include <exception>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "../detail/constants.hpp"
#include "../exceptions.hpp"
#include "../problem.hpp" // needed for cereal registration macro
#include "../types.hpp"

namespace pagmo
{

/// The CEC 2013 problems: Real-Parameter Single Objective Optimization Competition
/**
 * \image html cec2013.png
 * The 28 problems of the competition on real-parameter single objective optimization problems that
 * was organized for the 2013 IEEE Congress on Evolutionary Computation.
 *
 * **NOTE**: This set of UDAs require the data files that can be downloaded from the link below.
 * Upon construction, it expects to find two files named M_Dxx.txt and shift_data.txt in the folder indicated by
 * the constructor argument \p dir
 *
 * NOTE All problems are box-bounded, continuous, single objective problems.
 *
 * @see http://www.ntu.edu.sg/home/EPNSugan/index_files/CEC2013/CEC2013.htm
 * @see http://web.mysites.ntu.edu.sg/epnsugan/PublicSite/Shared%20Documents/CEC2013/cec13-c-code.zip
 */

class cec2013
{
public:
    /// Constructor
    /**
     * Will construct one of the 28 CEC2013 problems
     *
     * @param[in] prob_id The problem id. One of [1,2,...,28]
     * @param[in] dim problem dimension. One of [2,5,10,20,30,...,100]
     * @param[in] dir The path where the CEC2013 input files are located.
     *
     * **NOTE** Two files are expected to be in \p dir: "M_Dx.txt" and "shift_data.txt", where "x" is
     * the problem dimension
     *
     * @see http://web.mysites.ntu.edu.sg/epnsugan/PublicSite/Shared%20Documents/CEC2013/cec13-c-code.zip to find
     * the files
     *
     * @throws io_error if the files are not found
     */
    cec2013(unsigned int prob_id = 1u, unsigned int dim = 2u, const std::string &dir = "cec2013_data/") : m_prob_id(prob_id), m_y(dim), m_z(dim)
    {
        if (!(dim == 2u || dim == 5u || dim == 10u || dim == 20u || dim == 30u || dim == 40u || dim == 50u || dim == 60u
              || dim == 70u || dim == 80u || dim == 90u || dim == 100u)) {
            pagmo_throw(
                std::invalid_argument,
                "Error: CEC2013 Test functions are only defined for dimensions 2,5,10,20,30,40,50,60,70,80,90,100.");
        }

        std::string data_file_name(dir);
        // We create the full file name for the shift vector
        data_file_name.append("shift_data.txt");
        // And we read all data into m_origin_shift
        {
            std::ifstream data_file(data_file_name.c_str());
            if (!data_file.is_open()) {
                pagmo_throw(std::ios_base::failure,
                            std::string("Error: file not found. I was looking for ").append(data_file_name.c_str()));
            }
            std::istream_iterator<double> start(data_file), end;
            m_origin_shift = std::vector<double>(start, end);
            data_file.close();
        }

        // We create the full file name for the rotation matrix
        data_file_name = dir;
        data_file_name.append("M_D");
        data_file_name.append(std::to_string(dim));
        data_file_name.append(".txt");
        // And we read all datas into m_rotation_matrix
        {
            std::ifstream data_file(data_file_name.c_str());
            if (!data_file.is_open()) {
                pagmo_throw(std::ios_base::failure,
                            std::string("Error: file not found. I was looking for (") + data_file_name.c_str() + ")");
            }
            std::istream_iterator<double> start(data_file), end;
            m_rotation_matrix = std::vector<double>(start, end);
            data_file.close();
        }
    }
    /// Fitness computation
    /**
     * Computes the fitness for this UDP
     *
     * @param x the decision vector.
     *
     * @return the fitness of \p x.
     */
    vector_double fitness(const vector_double &x) const
    {
        return {0.};
    }
    /// Box-bounds
    /**
     * One of the optional methods of any user-defined problem (UDP).
     * It returns the box-bounds for this UDP.
     *
     * @return the lower and upper bounds for each of the decision vector components
     */
    std::pair<vector_double, vector_double> get_bounds() const
    {
        // all CEC 2013 problems have the same bounds
        vector_double lb(m_z.size(), -100.);
        vector_double ub(m_z.size(), 100.);
        return {lb, ub};
    }
    /// Problem name
    /**
     * One of the optional methods of any user-defined problem (UDP).
     *
     * @return a string containing the problem name
     */
    std::string get_name() const
    {
        return "CEC2013";
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
        ar(m_prob_id, m_rotation_matrix, m_origin_shift, m_y, m_z);
    }

private:
    // problem id
    unsigned int m_prob_id;
    // problem data
    std::vector<double> m_rotation_matrix;
    std::vector<double> m_origin_shift;

    // pre-allocated stuff for speed
    mutable std::vector<double> m_y;
    mutable std::vector<double> m_z;
};

} // namespace pagmo

PAGMO_REGISTER_PROBLEM(pagmo::cec2013)

#endif
