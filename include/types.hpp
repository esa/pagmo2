#ifndef PAGMO_TYPES_HPP
#define PAGMO_TYPES_HPP

#include <utility>
#include <vector>

namespace pagmo
{

using decision_vector = std::vector<double>;
using decision_vector_int = std::vector<long long>;
using fitness_vector = std::vector<double>;
using gradient_vector = std::vector<double>;
using sparsity_pattern = std::vector<std::pair<long,long>>;

}

#endif
