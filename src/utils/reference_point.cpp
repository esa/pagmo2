#include <algorithm>  // sample
#include <iterator>
#include <random>     // mt19937
#include <sstream>
#include <string>

#include <pagmo/utils/reference_point.hpp>
#include <pagmo/utils/multi_objective.hpp>  // perpendicular_distance


namespace pagmo{

ReferencePoint::ReferencePoint(size_t nobjs){
    coeffs.resize(nobjs);
    std::fill(coeffs.begin(), coeffs.end(), 0.0);
}

ReferencePoint::~ReferencePoint(){
    coeffs.clear();
}

double& ReferencePoint::operator[](int idx){
    return coeffs[idx];
}

size_t ReferencePoint::dim() const{
    return coeffs.size();
}

std::ostream& operator<<(std::ostream& ostr, const ReferencePoint& rp){
    std::ostringstream oss;
    oss << "[";
    std::copy(rp.coeffs.begin(), rp.coeffs.end()-1, std::ostream_iterator<double>(oss, ", ") );
    oss << rp.coeffs.back() << "]";
    ostr << oss.str();
    return ostr;
}

std::vector<ReferencePoint> generate_reference_point_level(
    ReferencePoint& rp,
    size_t remain,
    size_t level,
    size_t total
){
    std::vector<ReferencePoint> points;

    if(level == rp.dim()-1){
        rp[level] = 1.0*remain/total;
        points.push_back(rp);
    }else{
        for(size_t idx = 0; idx <= remain; idx++){
            rp[level] = 1.0*idx/total;
            auto np = generate_reference_point_level(rp, remain - idx, level + 1, total);
            points.reserve(points.size() + np.size());
            points.insert(points.end(), np.begin(), np.end());
        }
    }

    return points;
}

void ReferencePoint::add_candidate(size_t index, double distance){
    candidates.push_back(std::pair<size_t, double>(index, distance));
}

void ReferencePoint::remove_candidate(size_t index){
    for(size_t idx=0; idx<candidates.size(); idx++){
        if(candidates[idx].first == index){
            candidates.erase(candidates.begin() + idx);
        }
    }
}

void associate_with_reference_points(std::vector<ReferencePoint> &rps,
                                     std::vector<std::vector<double>> norm_objs,
                                     std::vector<std::vector<pop_size_t>> fronts){
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

size_t identify_niche_point(std::vector<ReferencePoint> &rps){
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
    std::vector<size_t> choice;
    std::sample(minimal_set.begin(), minimal_set.end(), std::back_inserter(choice),
                1, std::mt19937{std::random_device{}()});
    return choice[0];
}

// Section IV.E
int ReferencePoint::select_member() const{
    int selected = -1;
    if(candidate_count() != 0){
        if(member_count() == 0){  // Candidates but no members: rho == 0
            selected = nearest_candidate();
        }else{
            selected = random_candidate(); // Candidates and members: rho >= 1
        }
    }
    return selected;
}

int ReferencePoint::nearest_candidate() const{
    double min_dist = std::numeric_limits<double>::max();
    int min_idx = -1;
    for(size_t idx=0; idx<candidates.size(); idx++){
        if(candidates[idx].second < min_dist){
            min_dist = candidates[idx].second;
            min_idx  = candidates[idx].first;
        }
    }
    return min_idx;
}

int ReferencePoint::random_candidate() const{
    if(candidates.empty()){
        return -1;
    }
    std::vector<std::pair<size_t, double>> choice;
    std::sample(candidates.begin(), candidates.end(), std::back_inserter(choice),
                1, std::mt19937{std::random_device{}()});
    return choice[0].first;
}

} // namespace pagmo
