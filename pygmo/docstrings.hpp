#ifndef PYGMO_DOCSTRINGS_HPP
#define PYGMO_DOCSTRINGS_HPP

#include <string>

namespace pygmo
{

// Population docstrings.
std::string population_docstring();
std::string population_push_back_docstring();
std::string population_decision_vector_docstring();
std::string population_best_idx_docstring();
std::string population_worst_idx_docstring();
std::string population_size_docstring();
std::string population_set_xf_docstring();
std::string population_set_x_docstring();
std::string population_set_problem_seed_docstring();
std::string population_get_problem_docstring();
std::string population_get_f_docstring();
std::string population_get_x_docstring();
std::string population_get_ID_docstring();
std::string population_get_seed_docstring();

// Problem docstrings.
std::string problem_docstring();
std::string problem_fitness_docstring();
std::string problem_gradient_docstring();
std::string problem_get_best_docstring(const std::string &);

// Algorithm docstrings.
std::string algorithm_docstring();

// Exposed C++ problems docstrings.
std::string rosenbrock_docstring();

// Exposed C++ algorithms docstrings.
std::string cmaes_docstring();

}

#endif
