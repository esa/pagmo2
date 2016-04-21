#ifndef PAGMO_TYPES_HPP
#define PAGMO_TYPES_HPP

#include <utility>
#include <vector>

/// Root PaGMO namespace.
namespace pagmo
{

/// Alias for an <tt>std::vector</tt> of <tt>double</tt>s.
using vector_double = std::vector<double>;
/// Alias for an <tt>std::vector</tt> of <tt>std::pair</tt>s of the size type of pagmo::vector_double.
using sparsity_pattern = std::vector<std::pair<vector_double::size_type,vector_double::size_type>>;

} // namespaces

#endif
