#ifndef PAGMO_TYPES_HPP
#define PAGMO_TYPES_HPP

#include <utility>
#include <vector>
#include <boost/lexical_cast.hpp>

namespace pagmo
{

using decision_vector = std::vector<double>;
using fitness_vector = std::vector<double>;
using gradient_vector = std::vector<double>;
using sparsity_pattern = std::vector<std::pair<long,long>>;
using box_bounds = std::pair<decision_vector,decision_vector>;

}

// Give the possibility to disable stream overloads with appropriate #define.
#ifndef PAGMO_NO_STD_VECTOR_STREAM_OVERLOADS

#define PAGMO_MAX_OUTPUT_LENGTH 5
namespace std
{
	/// Overload stream insertion operator for std::vector<T>. It will only output the first
	/// PAGMO_MAX_OUTPUT_LENGTH elements.
	template < class T >
	inline ostream &operator<<(ostream &os, const vector<T> &v)
	{
		typename vector<T>::size_type len = v.size();
		if (len < PAGMO_MAX_OUTPUT_LENGTH) 
		{
			os << '[';
			for (typename std::vector<T>::size_type i = 0; i < v.size(); ++i) {
				os << boost::lexical_cast<std::string>(v[i]);
				if (i != v.size() - 1) {
					os << ", ";
				}
			}
			os << ']';
		} else {
			os << '[';
			for (typename std::vector<T>::size_type i = 0; i < PAGMO_MAX_OUTPUT_LENGTH; ++i) {
				os << boost::lexical_cast<std::string>(v[i]) << ", ";
			}
			os << " ... ]";
		}
		return os;
	}

	
}
#undef PAGMO_MAX_OUTPUT_LENGTH
#endif

#endif
