#ifndef PAGMO_DETAIL_SUPPORT_XEUS_CLING_HPP
#define PAGMO_DETAIL_SUPPORT_XEUS_CLING_HPP

#if defined(__CLING__)

#if __has_include(<nlohmann/json.hpp>)

#include <nlohmann/json.hpp>
#include <sstream>

namespace pagmo
{

namespace detail
{

template <typename T>
inline nlohmann::json cling_repr(const T &x)
{
    auto bundle = nlohmann::json::object();

    std::ostringstream oss;
    oss << x;
    bundle["text/plain"] = oss.str();

    return bundle;
}

} // namespace detail

} // namespace pagmo

#define PAGMO_IMPLEMENT_XEUS_CLING_REPR(name)                                                                               \
    namespace pagmo                                                                                                    \
    {                                                                                                                  \
    inline nlohmann::json mime_bundle_repr(const name &x)                                                              \
    {                                                                                                                  \
        return detail::cling_repr(x);                                                                                  \
    }                                                                                                                  \
    }

#else

// nlohmann::json not available.
#define PAGMO_IMPLEMENT_XEUS_CLING_REPR(name)

#endif

#else

// We are not using cling.
#define PAGMO_IMPLEMENT_XEUS_CLING_REPR(name)

#endif

#endif
