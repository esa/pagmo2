#include <pagmo/utils/hv_algos/hv_polar_approx.hpp>
#include <cmath>
#include <algorithm>

namespace pagmo
{


vector_double polar_approx::generate_random_ray(unsigned dim) const
{
    std::normal_distribution<double> dist(0.0, 1.0);
    
    vector_double ray(dim);
    double norm = 0;
    for (unsigned i = 0; i < dim; ++i) {
        ray[i] = std::abs(dist(m_e)); 
        norm += ray[i] * ray[i];
    }
    
    norm = std::sqrt(norm);
    for (unsigned i = 0; i < dim; ++i) {
        ray[i] /= norm;
    }
    return ray;
}

} // namespace pagmo
