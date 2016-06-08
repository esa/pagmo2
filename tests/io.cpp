#include "../include/io.hpp"

#define BOOST_TEST_MODULE io_test
#include <boost/test/included/unit_test.hpp>

#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

using namespace pagmo;
using namespace std::string_literals;

BOOST_AUTO_TEST_CASE(stream_print_test)
{
    // A few simple tests.
    std::ostringstream ss1, ss2;
    stream(ss1,1,2,3);
    ss2 << 1 << 2 << 3;
    BOOST_CHECK_EQUAL(ss1.str(),ss2.str());
    ss1.str("");
    ss2.str("");
    stream(ss1,"Hello "," world"s);
    ss2 << "Hello " << " world"s;
    BOOST_CHECK_EQUAL(ss1.str(),ss2.str());
    ss1.str("");
    ss2.str("");
    // Try with floating-point too.
    stream(ss1,1.234);
    ss2 << 1.234;
    BOOST_CHECK_EQUAL(ss1.str(),ss2.str());
    ss1.str("");
    ss2.str("");
    // Custom precision.
    ss1 << std::setprecision(10);
    ss2 << std::setprecision(10);
    stream(ss1,1.234);
    ss2 << 1.234;
    BOOST_CHECK_EQUAL(ss1.str(),ss2.str());
    ss1.str("");
    ss2.str("");
    // Special handling of bool.
    stream(ss1,true,' ',false);
    BOOST_CHECK_EQUAL(ss1.str(),"true false");
    ss1.str("");
    // Vectors.
    stream(ss1,std::vector<int>{});
    BOOST_CHECK_EQUAL(ss1.str(),"[]");
    ss1.str("");
    stream(ss1,std::vector<int>{1,2,3});
    ss2 << "[" << 1 << ", " << 2 << ", " << 3 << "]";
    BOOST_CHECK_EQUAL(ss1.str(),ss2.str());
    ss1.str("");
    ss2.str("");
    // Vector larger than the print limit.
    stream(ss1,std::vector<int>{1,2,3,4,5,6});
    ss2 << "[" << 1 << ", " << 2 << ", " << 3 << ", " << 4 << ", " << 5 << ", ... ]";
    BOOST_CHECK_EQUAL(ss1.str(),ss2.str());
    // Go for the print as well, yay.
    print(std::vector<int>{1,2,3,4,5,6});
}
