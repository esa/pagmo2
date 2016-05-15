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
#include <utility>

#include "../include/exceptions.hpp"
#include "../include/problem.hpp"
#include "../include/problems/hock_schittkowsky_71.hpp"
#include "../include/types.hpp"
#include "common_utils.hpp"
#include "prob_inner_python.hpp"
#include "pybind11.hpp"

namespace py = pybind11;
using namespace pagmo;

PYBIND11_PLUGIN(_core)
{
    py::module m("_core", "PyGMO's core module");

    py::class_<problem> problem_class(m,"problem");

    // Expose the generic problem interface.
    problem_class.def(py::init<const problem &>())
        .def("fitness",[](const problem &p, py::array_t<double,py::array::c_style> dv) {
            return pygmo::vd_to_a(p.fitness(pygmo::a_to_vd(dv)));
        },"Fitness.", py::arg("dv"))
        .def("gradient",[](const problem &p, py::array_t<double,py::array::c_style> dv) {
            return pygmo::vd_to_a(p.gradient(pygmo::a_to_vd(dv)));
        },"Gradient.", py::arg("dv"))
        .def("has_gradient",&problem::has_gradient)
        .def("gradient_sparsity",[](const problem &p) {
            return pygmo::sp_to_a(p.gradient_sparsity());
        },"Gradient sparsity.")
        .def("hessians",[](const problem &p, py::array_t<double,py::array::c_style> dv) {
            const auto tmp = p.hessians(pygmo::a_to_vd(dv));
            std::vector<py::array_t<double,py::array::c_style>> retval;
            std::transform(tmp.begin(),tmp.end(),std::back_inserter(retval),[](const auto &v) {
                return pygmo::vd_to_a(v);
            });
            return retval;
        },"Hessians.", py::arg("dv"))
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
        .def("get_bounds",[](const problem &p) {
            auto tmp = p.get_bounds();
            return py::make_tuple(pygmo::vd_to_a(std::move(tmp.first)),pygmo::vd_to_a(std::move(tmp.second)));
        })
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
        .def("get_extra_info",&problem::get_extra_info)
        .def("__repr__",[](const problem &p) {
            std::stringstream oss;
            oss << p;
            return oss.str();
        });

    py::class_<hock_schittkowsky_71> hs71(m,"hock_schittkowsky_71");
    hs71.def(py::init<>());

    problem_class.def(py::init<hock_schittkowsky_71>());
    problem_class.def("_extract",[](const problem &p, const hock_schittkowsky_71 &) {
        auto ptr = p.extract<hock_schittkowsky_71>();
        if (!ptr) {
            pagmo_throw(std::runtime_error,std::string("cannot extract an instance of type '") +
                typeid(hock_schittkowsky_71).name() + "'");
        }
        return hock_schittkowsky_71(*ptr);
    });

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
