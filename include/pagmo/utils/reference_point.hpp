/*
 *  Defines a ReferencePoint type used by the NSGA-III algorithm
 *
 */

#ifndef PAGMO_UTILS_REFERENCE_POINT
#define PAGMO_UTILS_REFERENCE_POINT

#include <iostream>
#include <optional>
#include <string>
#include <sstream>
#include <vector>

#include <pagmo/detail/visibility.hpp>  // PAGMO_DLL_PUBLIC
#include <pagmo/population.hpp>         // pop_size_t


namespace pagmo{

class PAGMO_DLL_PUBLIC ReferencePoint{
    public:
        ReferencePoint(size_t nobj);
        ~ReferencePoint();
        size_t dim() const;
        double& operator[](int);
        friend PAGMO_DLL_PUBLIC std::ostream& operator<<(std::ostream& ostr, const ReferencePoint& rp);
        void increment_members(){ ++nmembers; }
        void decrement_members(){ --nmembers; }
        size_t member_count() const{ return nmembers; }
        void add_candidate(size_t, double);
        void remove_candidate(size_t index);
        size_t candidate_count() const{ return candidates.size(); }
        std::vector<double> get_coeffs(){ return coeffs; }
        std::optional<size_t> nearest_candidate() const;
        std::optional<size_t> random_candidate() const;
        std::optional<size_t> select_member() const;
    protected:
        std::vector<double> coeffs{0};
        size_t nmembers{0};
        std::vector<std::pair<size_t, double>> candidates;
};

std::vector<ReferencePoint> generate_reference_point_level(
    ReferencePoint& rp,
    size_t remain,
    size_t level,
    size_t total
);

void associate_with_reference_points(
    std::vector<ReferencePoint> &,          // Reference points
    std::vector<std::vector<double>>,       // Normalized objectives
    std::vector<std::vector<pop_size_t>>    // NDS Fronts
);

size_t identify_niche_point(std::vector<ReferencePoint> &);

size_t n_choose_k(unsigned, unsigned);

}  // namespace pagmo

#endif
