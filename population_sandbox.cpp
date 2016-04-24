#include <string>

#include "include/io.hpp"
#include "include/population2.hpp"
#include "include/problem.hpp"
#include "include/problems/hock_schittkowsky_71.hpp"

using namespace pagmo;

// N logN complexity
template <typename T>
std::vector<vector_double::size_type> get_best_idx(vector_double::size_type N, T fitness_comparison_operator) 
{
    if (N > fitness_comparison_operator.m_fits.size()) {
        pagmo_throw(std::invalid_argument,"Best " + std::to_string(N) + " individuals requested, while population has size: " + std::to_string(fitness_comparison_operator.m_fits.size()) );
    }
    std::vector<vector_double::size_type> retval(fitness_comparison_operator.m_fits.size());
    std::iota(retval.begin(), retval.end(), 0);
    std::sort(retval.begin(), retval.end(),  fitness_comparison_operator);
    retval.resize(N);
    return retval;
}

/** N complexity
template <typename T>
size_type get_best_idx(T fitness_comparison_operator)
{
    if (size() == 0) {
        pagmo_throw(std::invalid_argument,"The population is empty cannot extract a best individua");
    }
    std::vector<size_type> retval(size());
    std::iota(retval.begin(), retval.end(), 0);
    auto it = std::min_element(retval.begin(), retval.end(), [this, &fitness_comparison_operator] (size_type i, size_type j) {return fitness_comparison_operator(m_container[i].f,m_container[j].f);} );
    return std::distance(retval.begin(), it);
}
*/

struct so_fitness_comparison 
{
    so_fitness_comparison(const std::vector<vector_double> &fits) : m_fits (fits) {};
    bool operator() (vector_double::size_type idx1,vector_double::size_type idx2) { return (m_fits[idx1][0]<m_fits[idx2][0]);}
    std::vector<vector_double> m_fits;
};


int main()
{
    // Constructing a population
    problem prob{hock_schittkowsky_71{}};
    population pop{prob, 10};
    print(pop);
    print(get_best_idx(10,so_fitness_comparison(pop.get_f()) ));
    //print(pop.get_best_idx(so_fitness_comparison()));
    
}
