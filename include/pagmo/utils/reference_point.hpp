/*
 *  Defines a ReferencePoint type used by the NSGA-III algorithm
 *
 */

#ifndef PAGMO_UTILS_REFERENCE_POINT
#define PAGMO_UTILS_REFERENCE_POINT

#include <iostream>
#include <string>
#include <sstream>
#include <vector>

#include <pagmo/detail/visibility.hpp>  // PAGMO_DLL_PUBLIC


namespace pagmo{

class PAGMO_DLL_PUBLIC ReferencePoint{
    public:
        ReferencePoint(size_t nobj);
        ~ReferencePoint();
        size_t dim() const;
        double& operator[](int);
        friend PAGMO_DLL_PUBLIC std::ostream& operator<<(std::ostream& ostr, const ReferencePoint& rp);
    protected:
        std::vector<double> coeffs;
};

std::vector<ReferencePoint> generate_reference_point_level(
    ReferencePoint& rp,
    size_t remain,
    size_t level,
    size_t total
);

}  // namespace pagmo

#endif
