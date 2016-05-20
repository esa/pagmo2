#ifndef PYGMO_PROBLEM_EXPOSITION_SUITE_HPP
#define PYGMO_PROBLEM_EXPOSITION_SUITE_HPP

#include <algorithm>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "../include/io.hpp"
#include "../include/problem.hpp"
#include "../include/problems/translate.hpp"
#include "../include/serialization.hpp"
#include "../include/type_traits.hpp"
#include "common_utils.hpp"
#include "pybind11.hpp"

namespace pygmo
{

namespace py = pybind11;

// Expose the get_bounds() getter.
template <typename Prob>
inline void expose_get_bounds(py::class_<Prob> &c)
{
    c.def("get_bounds",[](const Prob &p) {
        auto tmp = p.get_bounds();
        return py::make_tuple(vd_to_a(std::move(tmp.first)),vd_to_a(std::move(tmp.second)));
    });
}

// Expose fitness calculation.
template <typename Prob>
inline void expose_fitness(py::class_<Prob> &c)
{
    c.def("fitness",[](const Prob &p, py::array_t<double,py::array::c_style> dv) {
        return vd_to_a(p.fitness(a_to_vd(dv)));
    },"Fitness.", py::arg("dv"));
}

// Expose getter for number of equality constraints.
template <typename Prob, typename std::enable_if<pagmo::has_e_constraints<Prob>::value,int>::type = 0>
inline void expose_get_nec(py::class_<Prob> &c)
{
    c.def("get_nec",[](const Prob &p) {
        return p.get_nec();
    },"Number of equality constraints.");
}

template <typename Prob, typename std::enable_if<!pagmo::has_e_constraints<Prob>::value,int>::type = 0>
inline void expose_get_nec(py::class_<Prob> &)
{}

// Expose getter for number of inequality constraints.
template <typename Prob, typename std::enable_if<pagmo::has_i_constraints<Prob>::value,int>::type = 0>
inline void expose_get_nic(py::class_<Prob> &c)
{
    c.def("get_nic",[](const Prob &p) {
        return p.get_nic();
    },"Number of inequality constraints.");
}

template <typename Prob, typename std::enable_if<!pagmo::has_i_constraints<Prob>::value,int>::type = 0>
inline void expose_get_nic(py::class_<Prob> &)
{}

// Expose a constructor of pagmo::translate from Prob.
template <typename Prob>
inline void expose_translate_ctor(py::class_<pagmo::translate> &translate)
{
    // NOTE: need to do it like this in order to accept numpy arrays as input.
    translate.def("__init__",[](pagmo::translate &pt, const Prob &p, py::array_t<double,py::array::c_style> t) {
        // NOTE: in case of exceptions, pybind11 considers the object as not-constructed. It is important
        // that in custom __init__ methods like this we don't throw any exception after constructing the object,
        // or the we call explicitly the dtor before re-throwing.
        ::new (&pt) pagmo::translate(p,a_to_vd(t));
    },"Constructor from problem and translation vector.",py::arg("problem"),py::arg("translation"));
}

// Expose streaming. Use the C++ streaming, if available, otherwise generate it.
template <typename Prob, typename std::enable_if<std::is_base_of<std::ostream,
    std::decay_t<decltype(std::declval<std::ostream &>() << std::declval<const Prob &>())>>::value,int>::type = 0>
inline void expose_problem_repr(py::class_<Prob> &c, const std::string &)
{
    c.def("__repr__",[](const Prob &p) {
        std::ostringstream oss;
        oss << p;
        return oss.str();
    });
}

template <typename Prob, typename ... Args>
inline void expose_problem_repr(py::class_<Prob> &c, const std::string &name, const Args & ...)
{
    c.def("__repr__",[name](const Prob &) {
        using pagmo::stream;
        std::ostringstream oss;
        oss << "The exposed '" << name << "' C++ problem.\n\n";
        stream(oss,"Problem properties:\n");
        stream(oss,"-------------------\n");
        stream(oss,"    Has equality constraints  : ",pagmo::has_e_constraints<Prob>::value,'\n');
        stream(oss,"    Has inequality constraints: ",pagmo::has_i_constraints<Prob>::value,'\n');
        stream(oss,"    Has gradient              : ",pagmo::has_gradient<Prob>::value,'\n');
        stream(oss,"    Has Hessians              : ",pagmo::has_hessians<Prob>::value,'\n');
        return oss.str();
    });
}

// Expose gradient.
template <typename Prob, typename std::enable_if<pagmo::has_gradient<Prob>::value,int>::type = 0>
inline void expose_gradient(py::class_<Prob> &c)
{
    c.def("gradient",[](const Prob &p, py::array_t<double,py::array::c_style> dv) {
        return vd_to_a(p.gradient(a_to_vd(dv)));
    },"Gradient.", py::arg("dv"));
}

template <typename Prob, typename std::enable_if<!pagmo::has_gradient<Prob>::value,int>::type = 0>
inline void expose_gradient(py::class_<Prob> &)
{}

// Expose Hessians.
template <typename Prob, typename std::enable_if<pagmo::has_hessians<Prob>::value,int>::type = 0>
inline void expose_hessians(py::class_<Prob> &c)
{
    c.def("hessians",[](const Prob &p, py::array_t<double,py::array::c_style> dv) {
        const auto tmp = p.hessians(a_to_vd(dv));
        std::vector<py::array_t<double,py::array::c_style>> retval;
        std::transform(tmp.begin(),tmp.end(),std::back_inserter(retval),[](const auto &v) {
            return vd_to_a(v);
        });
        return retval;
    },"Hessians.", py::arg("dv"));
}

template <typename Prob, typename std::enable_if<!pagmo::has_hessians<Prob>::value,int>::type = 0>
inline void expose_hessians(py::class_<Prob> &)
{}

// Generic problem exposition. This function will expose:
// - default and copy constructor,
// - streaming,
// - the mandatory problem methods (fitness, bounds, nobj),
// - the optional methods,
// - constructors of pagmo::problem and pagmo::translate from Prob.
template <typename Prob>
inline py::class_<Prob> expose_problem(py::module &m, const std::string &name, py::class_<pagmo::problem> &problem_class,
    py::class_<pagmo::translate> *t_ptr = nullptr)
{
    // Initial exposition.
    py::class_<Prob> c(m,name.c_str());
    // Expose default ctor.
    c.def(py::init<>(),"Default constructor.");
    // Copy ctor.
    c.def(py::init<const Prob &>(),"Copy constructor.");
    // Repr.
    expose_problem_repr(c,name);

    // Let's first do the methods which are mandatory.
    // Fitness.
    expose_fitness(c);
    // Bounds.
    expose_get_bounds(c);
    // Number of objectives.
    c.def("get_nobj",[](const Prob &p) {
        return p.get_nobj();
    });

    // Now the optional methods.
    expose_get_nec(c);
    expose_get_nic(c);
    expose_gradient(c);
    expose_hessians(c);

    // Expose a constructor of pagmo::problem from Problem.
    problem_class.def(py::init<Prob>());
    // Expose a constructor of pagmo::translate from Problem.
    if (t_ptr != nullptr) {
        expose_translate_ctor<Prob>(*t_ptr);
    }

    // Serialization.
    c.def("__getstate__", [](const Prob &p) {
        // Serialize into JSON.
        std::ostringstream oss;
        {
        cereal::JSONOutputArchive oarchive(oss);
        oarchive(p);
        }
        // Return a tuple that fully encodes the state of the object.
        return py::make_tuple(oss.str());
    });
    c.def("__setstate__", [](Prob &p, py::tuple t) {
        // NOTE: the __setstate__ method is called by the serialization machinery
        // in Python after calling new(). The important bit here is that p has not been
        // constructed yet when this method is used by the pickle routines.
        // NOTE: it is thus *very important* that this method is never called explicitly
        // by any user code.
        if (t.size() != 1u) {
            // Object not constructed yet, ok to throw.
            pagmo_throw(std::runtime_error,"invalid problem state");
        }
        std::string tmp = t[0].cast<std::string>();
        // Default-construct the problem.
        ::new (&p) Prob();
        try {
            // From now on we need to be careful, as any exception thrown here
            // will mark the object as non-constructed, even if we indeed called the constructor
            // above. We need to catch any exception, call the problem dtor, and then re-throw it.
            std::istringstream iss;
            iss.str(tmp);
            {
                cereal::JSONInputArchive iarchive(iss);
                iarchive(p);
            }
        } catch (...) {
            p.~Prob();
            throw;
        }
    });

    return c;
}

}

#endif
