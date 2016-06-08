#ifndef PYGMO_DOCSTRINGS_HPP
#define PYGMO_DOCSTRINGS_HPP

#include <string>

namespace pygmo
{

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

std::string problem_docstring();
std::string get_best_docstring(const std::string &);

std::string algorithm_docstring();

std::string rosenbrock_docstring();

std::string cmaes_docstring();

}

#endif
