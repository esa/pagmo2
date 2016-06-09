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
#include "external/cereal/types/tuple.hpp"

#if defined(__clang__) || defined(__GNUC__)
    #pragma GCC diagnostic pop
#endif

#include <cstddef>
#include <random>
#include <sstream>
#include <string>
#ifdef PAGMO_ENABLE_EIGEN3
    #include <Eigen/Dense>
#endif


namespace cereal
{
// Implement serialization for the Mersenne twister engine.
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

#ifdef PAGMO_ENABLE_EIGEN3
    // Implement the serialization of the Eigen::Matrix class
    template <class Archive, class S, int R, int C> inline
    void CEREAL_SAVE_FUNCTION_NAME(Archive &ar, Eigen::Matrix<S,R,C> const &cb)
    {
        // Let's first save the dimension
        auto nrows = cb.rows();
        auto ncols = cb.cols();
        ar << nrows;
        ar << ncols;
        //And then the numbers
        for (decltype(nrows) i = 0; i < nrows; ++i) {
            for (decltype(nrows) j = 0; j < ncols; ++j) {
                ar << cb(i,j);
            }
        }
    }
    template <class Archive, class S, int R, int C> inline
    void CEREAL_LOAD_FUNCTION_NAME(Archive &ar, Eigen::Matrix<S,R,C> &cb)
    {
        decltype(cb.rows()) nrows;
        decltype(cb.cols()) ncols;
        // Let's first restore the dimension
        ar >> nrows;
        ar >> ncols;
        cb.resize(nrows,ncols);
        //And then the numbers
        for (decltype(nrows) i = 0; i < nrows; ++i) {
            for (decltype(nrows) j = 0; j < ncols; ++j) {
                ar >> cb(i,j);
            }
        }
    }
#endif
}

#endif
