#ifndef PAGMO_TYPES_HPP
#define PAGMO_TYPES_HPP

#include <atomic>
#include <vector>

namespace pagmo
{

using decision_vector = std::vector<double>;
using fitness_vector = std::vector<double>;
using gradient_vector = std::vector<double>;
using sparsity_pattern = std::vector<std::pair<long,long>>;
using atomic_counter = std::atomic<unsigned int long long>;


} // namespaces

#endif

