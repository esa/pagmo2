#ifndef PAGMO_SERIALIZATION_HPP
#define PAGMO_SERIALIZATION_HPP

#if defined(__clang__) || defined(__GNUC__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wconversion"
    #pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
    #pragma GCC diagnostic ignored "-Wdeprecated"
#endif

#include "external/cereal/archives/binary.hpp"
#include "external/cereal/archives/json.hpp"
#include "external/cereal/archives/portable_binary.hpp"
#include "external/cereal/types/base_class.hpp"
#include "external/cereal/types/common.hpp"
#include "external/cereal/types/memory.hpp"
#include "external/cereal/types/polymorphic.hpp"
#include "external/cereal/types/utility.hpp"
#include "external/cereal/types/vector.hpp"

#if defined(__clang__) || defined(__GNUC__)
    #pragma GCC diagnostic pop
#endif

#include <cstddef>
#include <random>
#include <sstream>
#include <string>

// Implement serialization for the Mersenne twister engine.
namespace cereal
{

template <class Archive, class UIntType,
    std::size_t w, std::size_t n, std::size_t m, std::size_t r,
    UIntType a, std:: size_t u, UIntType d, std::size_t s,
    UIntType b, std::size_t t, UIntType c, std::size_t l, UIntType f> inline
void CEREAL_SAVE_FUNCTION_NAME( Archive & ar, std::mersenne_twister_engine<UIntType,w,n,m,r,a,u,d,s,b,t,c,l,f> const & e )
{
    std::ostringstream oss;
    // Use the "C" locale.
    oss.imbue(std::locale::classic());
    oss << e;
    ar(oss.str());
}

template <class Archive, class UIntType,
    std::size_t w, std::size_t n, std::size_t m, std::size_t r,
    UIntType a, std:: size_t u, UIntType d, std::size_t s,
    UIntType b, std::size_t t, UIntType c, std::size_t l, UIntType f> inline
void CEREAL_LOAD_FUNCTION_NAME( Archive & ar, std::mersenne_twister_engine<UIntType,w,n,m,r,a,u,d,s,b,t,c,l,f> & e )
{
    std::istringstream iss;
    // Use the "C" locale.
    iss.imbue(std::locale::classic());
    std::string tmp;
    ar(tmp);
    iss.str(tmp);
    iss >> e;
}

}

#endif
