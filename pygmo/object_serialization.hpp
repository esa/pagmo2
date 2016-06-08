#ifndef PYGMO_OBJECT_SERIALIZATION_HPP
#define PYGMO_OBJECT_SERIALIZATION_HPP

#include "python_includes.hpp"

#include <boost/numeric/conversion/cast.hpp>
#include <boost/python/import.hpp>
#include <boost/python/object.hpp>
#include <vector>

#include "../include/serialization.hpp"
#include "common_utils.hpp"

namespace cereal
{

template <typename Archive>
void save(Archive &archive, const boost::python::object &o)
{
    // NOTE: these functions can be improved by using cereal's binary data
    // construct, at least for binary archives. This would allow to avoid the
    // extra copy. That would require to save the length of the bytes
    // object as well.
    // http://stackoverflow.com/questions/27518554/c-cereal-serialize-c-style-array
    using namespace boost::python;
    // This will dump to a bytes object.
    object tmp = import("pickle").attr("dumps")(o);
    // This gives a null-terminated char * to the internal
    // content of the bytes object.
    auto ptr = PyBytes_AsString(tmp.ptr());
    if (!ptr) {
        pygmo_throw(PyExc_TypeError,"pickle dumps did not return a bytes object");
    }
    // NOTE: this will be the length of the bytes object *without* the terminator.
    const auto size = len(tmp);
    // NOTE: we store as char here because that's what is returned by the CPython function.
    // From Python it seems like these are unsigned chars, but this should not concern us.
    std::vector<char> v(ptr,ptr + size);
    archive(v);
}

template <typename Archive>
void load(Archive &archive, boost::python::object &o)
{
    using namespace boost::python;
    // Extract the char vector.
    std::vector<char> v;
    archive(v);
    auto b = pygmo::make_bytes(v.data(),boost::numeric_cast<Py_ssize_t>(v.size()));
    o = import("pickle").attr("loads")(b);
}

}

#endif
