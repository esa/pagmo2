// NOTE: the order of inclusion in the first two items here is forced by these two issues:
// http://mail.python.org/pipermail/python-list/2004-March/907592.html
// http://mail.python.org/pipermail/new-bugs-announce/2011-March/010395.html
#if defined(_WIN32)
#include <cmath>
#include <Python.h>
#else
#include <Python.h>
#include <cmath>
#endif

#include <algorithm>
#include <iterator>
#include <sstream>
#include <string>

#include "../include/exceptions.hpp"
#include "../include/problem.hpp"
#include "../include/problems/hock_schittkowsky_71.hpp"
#include "../include/problems/inventory.hpp"
#include "../include/problems/rosenbrock.hpp"
#include "../include/problems/translate.hpp"
#include "../include/problems/zdt.hpp"
#include "../include/rng.hpp"
#include "../include/types.hpp"
#include "common_utils.hpp"
#include "prob_inner_python.hpp"
#include "problem_exposition_suite.hpp"
#include "pybind11.hpp"

namespace py = pybind11;
using namespace pagmo;

PYBIND11_PLUGIN(_core)
{
    py::module m("_core", "PyGMO's core module");

    py::class_<problem> problem_class(m,"problem");

    // Expose the generic problem interface.
    problem_class.def(py::init<const problem &>())
        .def("has_gradient",&problem::has_gradient)
        .def("gradient_sparsity",[](const problem &p) {
            return pygmo::sp_to_a(p.gradient_sparsity());
        },"Gradient sparsity.")
        .def("has_hessians",&problem::has_hessians)
        .def("hessians_sparsity",[](const problem &p) {
            const auto tmp = p.hessians_sparsity();
            std::vector<py::array_t<pagmo::vector_double::size_type,py::array::c_style>> retval;
            std::transform(tmp.begin(),tmp.end(),std::back_inserter(retval),[](const auto &s) {
                return pygmo::sp_to_a(s);
            });
            return retval;
        },"Hessians sparsity.")
        .def("get_nobj",&problem::get_nobj)
        .def("get_nx",&problem::get_nx)
        .def("get_nf",&problem::get_nf)
        .def("get_nec",&problem::get_nec)
        .def("get_nic",&problem::get_nic)
        .def("get_nc",&problem::get_nc)
        .def("get_fevals",&problem::get_fevals)
        .def("get_gevals",&problem::get_gevals)
        .def("get_hevals",&problem::get_hevals)
        .def("get_gs_dim",&problem::get_gs_dim)
        .def("get_hs_dim",&problem::get_hs_dim)
        .def("set_seed",&problem::set_seed)
        .def("has_set_seed",&problem::has_set_seed)
        .def("is_stochastic",&problem::is_stochastic)
        .def("get_name",&problem::get_name)
        .def("get_extra_info",&problem::get_extra_info);
    // These are shared with the exposition of concrete C++ problems.
    pygmo::expose_problem_repr(problem_class,"");
    pygmo::expose_fitness(problem_class);
    pygmo::expose_gradient(problem_class);
    pygmo::expose_get_bounds(problem_class);

    // Expose the translate problem. We do it first because we will need to expose its constructors from
    // concrete C++ problems later.
    auto t_prob = pygmo::expose_problem<translate>(m,"translate",problem_class);
    // Expose a constructor of translate from translate. Translate-ception.
    pygmo::expose_translate_ctor<translate>(t_prob);

    // Exposition of concrete C++ problems.
    pygmo::expose_problem<hock_schittkowsky_71>(m,"hock_schittkowsky_71",problem_class,&t_prob);
    auto rb = pygmo::expose_problem<rosenbrock>(m,"rosenbrock",problem_class,&t_prob);
    rb.def(py::init<unsigned int>(),"Constructor from dimension.",py::arg("dim"));
    auto inv = pygmo::expose_problem<inventory>(m,"inventory",problem_class,&t_prob);
    // Here we define two separate constructors because if we default the seed to pagmo::random_device::next(),
    // then the seed default value will be always the same (that is, the value randomly selected when the exposition
    // code is running upon importing pygmo). These 2 ctors seem to achieve what we need from the inventory class.
    inv.def(py::init<unsigned,unsigned>(),"Constructor from weeks and sample size (seed is randomly-generated).",
        py::arg("weeks") = 4u,py::arg("sample_size") = 10u);
    inv.def(py::init<unsigned,unsigned,unsigned>(),"Constructor from weeks, sample size and seed.",
        py::arg("weeks") = 4u,py::arg("sample_size") = 10u, py::arg("seed"));
    auto zdt = pygmo::expose_problem<pagmo::zdt>(m,"zdt",problem_class,&t_prob);
    zdt.def(py::init<unsigned,unsigned>(),"Constructor from id and param.",
        py::arg("id") = 1u,py::arg("param") = 30u);

    // problem_class.def("_extract",[](const problem &p, const hock_schittkowsky_71 &) {
    //     auto ptr = p.extract<hock_schittkowsky_71>();
    //     if (!ptr) {
    //         pagmo_throw(std::runtime_error,std::string("cannot extract an instance of type '") +
    //             typeid(hock_schittkowsky_71).name() + "'");
    //     }
    //     return hock_schittkowsky_71(*ptr);
    // });

    // This needs to go last, as it needs to have the lowest precedence among all ctors.
    problem_class.def(py::init<py::object>());
    problem_class.def("_extract",[](const problem &p, py::object o) {
        auto ptr = p.extract<py::object>();
        if (!ptr || pygmo::type(*ptr) != pygmo::type(o)) {
            pagmo_throw(std::runtime_error,"cannot extract an instance of type '" +
                pygmo::str(pygmo::type(o)).cast<std::string>() + "'");
        }
        return pygmo::deepcopy(*ptr);
    });

    return m.ptr();
}
