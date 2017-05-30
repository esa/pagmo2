#define PY_ARRAY_UNIQUE_SYMBOL uda_basic_ARRAY_API
#include <pygmo/numpy.hpp>

#include <boost/python/module.hpp>

#include <pygmo/algorithm_exposition_suite.hpp>

#include "uda_basic.hpp"

BOOST_PYTHON_MODULE(uda_basic) {
  pygmo::numpy_import_array();

  pygmo::expose_algorithm<uda_basic>("uda_basic", "My basic UDA.");
}
