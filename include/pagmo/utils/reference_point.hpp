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
#include <pagmo/rng.hpp>                // random_engine_type


namespace pagmo{

class PAGMO_DLL_PUBLIC ReferencePoint{
    public:
        ReferencePoint(size_t nobj);
        size_t dim() const;
        //double& operator[](int);
        double& operator[](size_t);
        friend PAGMO_DLL_PUBLIC std::ostream& operator<<(std::ostream& ostr, const ReferencePoint& rp);
        void increment_members(){ ++nmembers; }
        void decrement_members(){ --nmembers; }
        size_t member_count() const{ return nmembers; }
        void add_candidate(size_t, double);
        void remove_candidate(size_t index);
        size_t candidate_count() const{ return candidates.size(); }
        const std::vector<double> &get_coeffs() const{ return coeffs; }
        std::optional<size_t> nearest_candidate() const;
        std::optional<size_t> random_candidate(detail::random_engine_type &) const;
        std::optional<size_t> select_member(detail::random_engine_type &) const;
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
    std::vector<ReferencePoint> &,                  // Reference points
    const std::vector<std::vector<double>> &,       // Normalized objectives
    const std::vector<std::vector<pop_size_t>> &    // NDS Fronts
);

size_t identify_niche_point(std::vector<ReferencePoint> &, detail::random_engine_type &);

size_t n_choose_k(size_t, size_t);

}  // namespace pagmo

#endif
