#define BOOST_TEST_MODULE eigen_serialization_test
#include <boost/test/included/unit_test.hpp>

#include <iterator>
#include <random>
#include <sstream>
#include <vector>
#include <Eigen/Dense>

#include "../include/serialization.hpp"

template <typename T>
void serialize_deserialize_and_compare(T &object)
{
    auto object_copy(object);
    std::stringstream ss;
    {
    cereal::JSONOutputArchive oarchive(ss);
    oarchive(object_copy);
    }
    // Change the content of p before deserializing.
    object_copy = T{};
    {
    cereal::JSONInputArchive iarchive(ss);
    iarchive(object_copy);
    }
    auto nrows = object.rows();
    auto ncols = object.cols();
    BOOST_CHECK_EQUAL(nrows, object_copy.rows());
    BOOST_CHECK_EQUAL(ncols, object_copy.cols());
    //And then we check
    for (decltype(nrows) i = 0; i < nrows; ++i) {
        for (decltype(nrows) j = 0; j < ncols; ++j) {
            BOOST_CHECK_EQUAL(object_copy(i,j), object(i,j));
        }
    }
}

BOOST_AUTO_TEST_CASE(matrix_serialization_test)
{
    {
    Eigen::Matrix3f m;
    m << 1, 2, 3, 4, 5, 6, 7, 8, 9;
    serialize_deserialize_and_compare(m);
    }
    {
    Eigen::Vector2d a(5.0, 6.0);
    Eigen::Vector3d b(5.0, 6.0, 7.0);
    Eigen::Vector4d c(5.0, 6.0, 7.0, 8.0);
    serialize_deserialize_and_compare(a);
    serialize_deserialize_and_compare(b);
    serialize_deserialize_and_compare(c);
    }
    {
    Eigen::MatrixXd a = Eigen::MatrixXd::Identity(8, 8); // auto cannot be used here as Eigen uses expression templates, thus auto Eigen::MatrixXd::Identity is not an Eigen::Matrix
    Eigen::VectorXd b = Eigen::VectorXd::Zero(10);       // auto cannot be used here for the same reason
    serialize_deserialize_and_compare(a);
    serialize_deserialize_and_compare(b);
    }
}
