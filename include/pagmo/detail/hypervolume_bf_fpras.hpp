namespace pagmo
{

/// Bringmann-Friedrich approximation method
/**
 * This class contains the implementation of the Bringmann-Friedrich approximation scheme (FPRAS),
 * reduced to a special case of approximating the hypervolume indicator.
 * @see "Approximating the volume of unions and intersections of high-dimensional geometric objects", Karl Bringmann,
 * Tobias Friedrich.
 *
 * @author Krzysztof Nowak (kn@kiryx.net)
 * @author Marcus M�rtens (mmarcusx@gmail.com)
 */

class bf_fpras : public hv_algorithm
{
public:
    /// Constructor
    /**
    * Constructs an instance of the algorithm
    *
    * @param[in] eps accuracy of the approximation
    * @param[in] delta confidence of the approximation
    * @param[in] seed Seeding for the pseudo-random number generator
    */
    bf_fpras(const double eps, const double delta, unsigned int seed = pagmo::random_device::next())
        : m_eps(eps), m_delta(delta), m_e(seed)
    {
    }

    /// Verify before compute
    /**
    * Verifies whether given algorithm suits the requested data.
    *
    * @param[in] points vector of points containing the d dimensional points for which we compute the hypervolume
    * @param[in] r_point reference point for the vector of points
    *
    * @throws value_error when trying to compute the hypervolume for the non-maximal reference point
    */
    void verify_before_compute(const std::vector<vector_double> &points, const vector_double &r_point) const
    {
        hv_algorithm::assert_minimisation(points, r_point);
    }

    /// Compute method
    /**
    * Compute the hypervolume using FPRAS.
    *
    * @see "Approximating the volume of unions and intersections of high-dimensional geometric objects", Karl Bringmann,
    * Tobias Friedrich.
    *
    * @param[in] points vector of fitness_vectors for which the hypervolume is computed
    * @param[in] r_point distinguished "reference point".
    *
    * @return approximated hypervolume
    */
    double compute(std::vector<vector_double> &points, const vector_double &r_point) const
    {
        auto n = points.size();
        auto dim = r_point.size();

        // We do not want to continue if the floating point operations on eps and delta result in NaNs
        if (!(std::isfinite(12. * std::log(1. / m_delta) / std::log(2.) * n / m_eps / m_eps))) {
            pagmo_throw(std::invalid_argument, "Check the parameters of your call. There was NaN detected.");
        }
        boost::uint_fast64_t T
            = static_cast<boost::uint_fast64_t>(12. * std::log(1. / m_delta) / std::log(2.) * n / m_eps / m_eps);

        // Partial sums of consecutive boxes
        vector_double sums(n, 0.0);

        // points iterator
        std::vector<vector_double>::iterator it_p;

        // volume iterator - used for finding the contributor using std::lower_bound
        vector_double::iterator it_sums;

        unsigned int i = 0u;

        // Total sum of every box
        double V = 0.0;
        for (it_p = points.begin(); it_p != points.end(); ++it_p) {
            V = (sums[i++] = V + hv_algorithm::volume_between(*it_p, r_point));
        }

        unsigned long long M = 0;     // Round counter
        unsigned long long M_sum = 0; // Total number of samples over every round so far

        vector_double rnd_point(dim, 0.0); // Container for the random point
        auto unireal_dist = std::uniform_real_distribution<double>(0.0, 1.0);

        while (true) {
            // Get the random volume in-between [0, V] range, in order to choose the box with probability sums[i] / V

            auto V_dist = std::uniform_real_distribution<double>(0.0, V);
            auto r = V_dist(m_e);

            // Find the contributor using binary search
            it_sums = std::lower_bound(sums.begin(), sums.end(), r);
            i = static_cast<unsigned int>(std::distance(sums.begin(), it_sums));

            // Sample a point inside the 'box' (r_point, points[i])
            for (unsigned int d_idx = 0u; d_idx < dim; ++d_idx) {
                rnd_point[d_idx] = (points[i][d_idx] + unireal_dist(m_e) * (r_point[d_idx] - points[i][d_idx]));
            }

            unsigned int j = 0u;
            do {
                if (M_sum >= T) {
                    return (T * V) / static_cast<double>(n * M);
                }
                j = static_cast<unsigned int>(n * unireal_dist(m_e));
                ++M_sum;
            } while (!(hv_algorithm::dom_cmp(rnd_point, points[j], 0) == hv_algorithm::DOM_CMP_B_DOMINATES_A));
            ++M;
        }
    }

    /// Exclusive method
    /**
    * This algorithm does not support this method.
    */
    double exclusive(const unsigned int p_idx, std::vector<vector_double> &points, const vector_double &r_point) const
    {
        (void)p_idx;
        (void)points;
        (void)r_point;
        pagmo_throw(std::invalid_argument, "This method is not supported by the bf_fpras algorithm");
    }

    /// Least contributor method
    /**
    * This algorithm does not support this method.
    */
    unsigned int least_contributor(std::vector<vector_double> &points, const vector_double &r_point) const
    {
        (void)points;
        (void)r_point;
        pagmo_throw(std::invalid_argument, "This method is not supported by the bf_fpras algorithm");
    }

    /// Greatest contributor method
    /**
    * This algorithm does not support this method.
    */
    unsigned int greatest_contributor(std::vector<vector_double> &points, const vector_double &r_point) const
    {
        (void)points;
        (void)r_point;
        pagmo_throw(std::invalid_argument, "This method is not supported by the bf_fpras algorithm");
    }

    /// Contributions method
    /**
    * As of yet, this algorithm does not support this method, even in its naive form, due to a poor handling of the
    * dominated points.
    */
    vector_double contributions(std::vector<vector_double> &points, const vector_double &r_point) const
    {
        (void)points;
        (void)r_point;
        pagmo_throw(std::invalid_argument, "This method is not supported by the bf_fpras algorithm");
    }

    /// Clone method.
    std::shared_ptr<hv_algorithm> clone() const
    {
        return std::shared_ptr<hv_algorithm>(new bf_fpras(*this));
    }

    /// Algorithm name
    std::string get_name() const
    {
        return "bf_fpras algorithm";
    }

private:
    // error of the approximation
    const double m_eps;
    // probabiltiy of error
    const double m_delta;

    mutable detail::random_engine_type m_e;
};
}
