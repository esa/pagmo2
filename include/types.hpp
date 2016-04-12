#ifndef PAGMO_TYPES_HPP
#define PAGMO_TYPES_HPP

#include <vector>

namespace pagmo
{

using vector_double = std::vector<double>;
using sparsity_pattern = std::vector<std::pair<vector_double::size_type,vector_double::size_type>>;

} // namespaces

#endif

