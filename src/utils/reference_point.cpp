#include <iterator>
#include <sstream>
#include <string>

#include <pagmo/utils/reference_point.hpp>


namespace pagmo{

ReferencePoint::ReferencePoint(size_t nobjs){
    coeffs.resize(nobjs);
    std::fill(coeffs.begin(), coeffs.end(), 0.0);
}

ReferencePoint::~ReferencePoint(){
    coeffs.clear();
}

double& ReferencePoint::operator[](int idx){
    return coeffs[idx];
}

size_t ReferencePoint::dim() const{
    return coeffs.size();
}

std::ostream& operator<<(std::ostream& ostr, const ReferencePoint& rp){
    std::ostringstream oss;
    oss << "(";
    std::copy(rp.coeffs.begin(), rp.coeffs.end()-1, std::ostream_iterator<double>(oss, ", ") );
    oss << rp.coeffs.back() << ")";
    ostr << oss.str();
    return ostr;
}

std::vector<ReferencePoint> generate_reference_point_level(
    ReferencePoint& rp,
    size_t remain,
    size_t level,
    size_t total
){
    std::vector<ReferencePoint> points;

    if(level == rp.dim()-1){
        rp[level] = 1.0*remain/total;
        points.push_back(rp);
    }else{
        for(size_t idx = 0; idx <= remain; idx++){
            rp[level] = 1.0*idx/total;
            auto np = generate_reference_point_level(rp, remain - idx, level + 1, total);
            points.reserve(points.size() + np.size());
            points.insert(points.end(), np.begin(), np.end());
        }
    }

    return points; // Trust the elision
}

} // namespace pagmo
