#include <boost/python/module.hpp>

#include <pygmo/problem_exposition_suite.hpp>
#include <pygmo/register_ap.hpp>

#include "udp_basic.hpp"

BOOST_PYTHON_MODULE(udp_basic)
{
    pygmo::register_ap();

    pygmo::expose_problem<udp_basic>("udp_basic", "My basic UDP.");
}
