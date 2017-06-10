#include <boost/python/module.hpp>

#include <pygmo/algorithm_exposition_suite.hpp>
#include <pygmo/register_ap.hpp>

#include "uda_basic.hpp"

BOOST_PYTHON_MODULE(uda_basic)
{
    pygmo::register_ap();

    pygmo::expose_algorithm<uda_basic>("uda_basic", "My basic UDA.");
}
