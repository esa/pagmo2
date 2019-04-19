#include <vector>
#include <iostream>
#include <algorithm> // std::shuffle, std::transform
#include <iomanip>
#include <numeric> // std::iota, std::inner_product
#include <random>
#include <string>
#include <tuple>
#include <pagmo/algorithm.hpp> // needed for the cereal macro
#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/problems/decompose.hpp>
#include <pagmo/rng.hpp>
class preference
{
public:
	//std::vector < double > ind1;
	preference()
		//	:ind1 ( ind)
	{
		//ind = ind1;
	};
	std::vector <double> linear(const std::vector < double > &ind1);
	~preference();
private:
	std::vector < double > ind1;
};

class linearPreference : public preference
{
public:
	std::vector < double > weights;
	//int linearPreference(int n)
	//	//	:ind1(ind);
	//	
	//{
	//	for (int i = 0; i < n; i++) {
	//		weights.push_back(1 / n);
	//	}
	//	//ind = ind1;
	//};
	std::vector <double> utility(const std::vector<std::vector<double>> &pop) {
		int n = pop[0].size;
		std::vector<double> utility(n);
		for (auto idx : pop) {

			for (int i = 0; i < n; i++) {
				weights.push_back(1 / n);
				utility[i] = weights[i] * ind1[i];
			}
		}
		return utility;
	};
	std::vector <unsigned int> rank(const std::vector<double> &utility) {
		int M = utility.size;
		std::vector<vector_double::size_type> indexes(M);
		std::iota(indexes.begin(), indexes.end(), vector_double::size_type(0u));
		std::sort(indexes.begin(), indexes.end(),
			[&utility](vector_double::size_type idx1, vector_double::size_type idx2) {
			return detail::less_than_f(utility[idx1][0], utility[idx2][0]);
		});
		//retval[indexes[0]] = std::numeric_limits<double>::infinity();
		//retval[indexes[N - 1u]] = std::numeric_limits<double>::infinity();
		//double df = non_dom_front[indexes[N - 1u]][i] - non_dom_front[indexes[0]][i];
		//for (decltype(N - 2u) j = 1u; j < N - 1u; ++j) {
		//	retval[indexes[j]] += (non_dom_front[indexes[j + 1u]][i] - non_dom_front[indexes[j - 1u]][i]) / df;
		//}
	return indexes;
	}
	
	//~linearPreference();
private:
	std::vector < double > ind1;
};


