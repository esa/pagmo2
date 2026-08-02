/* Copyright 2017-2021 PaGMO development team

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

#include <algorithm>
#include <iterator>
#include <limits>
#include <optional>
#include <sstream>
#include <string>

#include <pagmo/detail/nsga3_impl.hpp>  // perpendicular_distance, choose_random_element
#include <pagmo/detail/reference_point.hpp>


namespace pagmo{

namespace detail{

reference_point::reference_point(size_t nobjs){
    coeffs.resize(nobjs);
    std::fill(coeffs.begin(), coeffs.end(), 0.0);
}

double& reference_point::operator[](size_t idx){
    return coeffs[idx];
}

size_t reference_point::dim() const{
    return coeffs.size();
}

std::ostream& operator<<(std::ostream& ostr, const reference_point& rp){
    std::ostringstream oss;
    oss << "[";
    std::copy(rp.coeffs.begin(), rp.coeffs.end()-1, std::ostream_iterator<double>(oss, ", ") );
    oss << rp.coeffs.back() << "]";
    ostr << oss.str();
    return ostr;
}

std::vector<reference_point> generate_reference_point_level(
    reference_point& rp,
    size_t remain,
    size_t level,
    size_t total
){
    std::vector<reference_point> points;

    if(level == rp.dim()-1){
        rp[level] = static_cast<double>(remain)/static_cast<double>(total);
        points.push_back(rp);
    }else{
        for(size_t idx = 0; idx <= remain; idx++){
            rp[level] = static_cast<double>(idx)/static_cast<double>(total);
            auto np = generate_reference_point_level(rp, remain - idx, level + 1, total);
            points.reserve(points.size() + np.size());
            points.insert(points.end(), np.begin(), np.end());
        }
    }

    return points;
}

std::vector<reference_point> generate_uniform_reference_points(size_t nobjs, size_t divisions){
    reference_point rp(nobjs);
    return generate_reference_point_level(rp, divisions, 0, divisions);
}

void reference_point::add_candidate(size_t index, double distance){
    candidates.push_back(std::pair<size_t, double>(index, distance));
}

void reference_point::remove_candidate(size_t index){
    for(size_t idx=0; idx<candidates.size(); idx++){
        if(candidates[idx].first == index){
            candidates.erase(candidates.begin() + static_cast<std::vector<std::pair<size_t, double>>::difference_type>(idx));
            break;  // Candidate indices are unique
        }
    }
}

void associate_with_reference_points(std::vector<reference_point> &rps,
                                     const std::vector<std::vector<double>> &norm_objs,
                                     const std::vector<std::vector<pop_size_t>> &fronts){
    for(size_t f=0; f<fronts.size(); f++){
        for(size_t i=0; i<fronts[f].size(); i++){
            size_t nearest = 0;
            double min_dist = std::numeric_limits<double>::max();
            for(size_t p=0; p<rps.size(); p++){
                double dist = perpendicular_distance(rps[p].get_coeffs(), norm_objs[fronts[f][i]]);
                if(dist < min_dist){
                    min_dist = dist;
                    nearest = p;
                }
            }
            if(f != fronts.size()-1){
                rps[nearest].increment_members();
            }else{
                rps[nearest].add_candidate(fronts[f][i], min_dist);
            }
        }
    }
}

size_t identify_niche_point(std::vector<reference_point> &rps, random_engine_type &reng){
    size_t min_size = std::numeric_limits<size_t>::max();
    std::vector<size_t> minimal_set;
    for(const auto &rp: rps){
        min_size = std::min(min_size, rp.member_count());
    }
    for(size_t idx=0; idx<rps.size(); idx++){
        if(rps[idx].member_count() == min_size){
            minimal_set.push_back(idx);
        }
    }
    // Return a random element from the minimal set
    return choose_random_element<size_t>(minimal_set, reng);
}

// Section IV.E
std::optional<size_t> reference_point::select_member(random_engine_type &reng) const{
    std::optional<size_t> selected = std::nullopt;
    if(candidate_count() != 0){
        if(member_count() == 0){  // Candidates but no members: rho == 0
            selected = nearest_candidate();
        }else{
            selected = random_candidate(reng); // Candidates and members: rho >= 1
        }
    }
    return selected;
}

std::optional<size_t> reference_point::nearest_candidate() const{
    double min_dist = std::numeric_limits<double>::max();
    std::optional<size_t> min_idx = std::nullopt;
    for(size_t idx=0; idx<candidates.size(); idx++){
        if(candidates[idx].second < min_dist){
            min_dist = candidates[idx].second;
            min_idx  = candidates[idx].first;
        }
    }
    return min_idx;
}

std::optional<size_t> reference_point::random_candidate(random_engine_type &reng) const{
    if(candidates.empty()){
        return std::nullopt;
    }
    return choose_random_element<std::pair<size_t, double>>(candidates, reng).first;
}

size_t n_choose_k(size_t n, size_t k){
    if(k == 0){
        return 1u;
    }
    return n*n_choose_k(n-1, k-1)/k;
}

} // namespace detail

} // namespace pagmo
