#ifndef PAGMO_UTIL_HV_ALGORITHMS_HV2D_H
#define PAGMO_UTIL_HV_ALGORITHMS_HV2D_H

#include <set>

#include "../utils/hv_algorithms/hv_algorithm.hpp"

namespace pagmo {

    class hypervolume
    {
    public:
    	/// Default constructor
    	/**
    	* Initiates hypervolume with empty set of points.
    	* Used for serialization purposes.
    	*/
    	hypervolume() : m_copy_points(true), m_verify(false)
    	{
    		m_points.resize(0);
    	}

    	/// Constructor from initializer list
    	/**
    	*  Constructs a hypervolume object from an initializer list, provided by curly-braces syntax
    	*  @example	hypervolume hv{{2,3},{3,4}};
    	*
    	*/
    	hypervolume(std::initializer_list<std::vector<double>> points) :m_points(points), m_copy_points(true), m_verify(true) {
            if (m_verify) {
    			verify_after_construct();
    		}
    	}

    	/// Constructor from a vector of points and a bool
    	/**
    	* Constructs a hypervolume object from a provided set of points.
    	*
    	* @param[in] points vector of points for which the hypervolume is computed
    	* @param[in] verify flag stating whether the points should be verified after the construction.
    	*			 This turns off the validation for the further computation as well, use 'set_verify'
    	*			 flag to alter it later.
    	*/
    	hypervolume(const std::vector<vector_double> &points, const bool verify) : m_points(points), m_copy_points(true), m_verify(verify)
    	{
    		if (m_verify) {
    			verify_after_construct();
    		}
    	}

    	/// Copy constructor.
    	/**
    	* Will perform a deep copy of hypervolume object
    	*
    	* @param[in] hv hypervolume object to be copied
    	*/
    	hypervolume(const hypervolume &hv) = default;

    	/// Constructor from population
    	/**
    	* Constructs a hypervolume object, where points are elicited from the referenced population object.
    	*
    	* @param[in] reference parameter for the population object
    	* @param[in] verify flag stating whether the points should be verified after the construction.
    	*			 This turns off the validation for the further computation as well, use 'set_verify' flag to alter it later.
    	*/
    	hypervolume(const pagmo::population &pop, const bool verify) : m_copy_points(true), m_verify(verify)
    	{
    		if (pop.get_problem().get_nc() == 0) {
    			m_points = pop.get_f();
    		} else {
    			pagmo_throw(std::invalid_argument, "The problem of the population is not unconstrained. Only unconstrained populations can be used to constructs hypervolumes.");
    		}
    		if (m_verify) {
    			verify_after_construct();
    		}
    	}


    	/// Setter for 'copy_points' flag
    	/**
    	* Sets the hypervolume as a single use object.
    	* It is used in cases where we are certain that we can alter the original set of points from the hypervolume object.
    	* This is useful when we don't want to make a copy of the points first, as most algorithms alter the original set.
    	*
    	* This may result in unexpected behaviour when used incorrectly (e.g. requesting the computation twice out of the same object)
    	*
    	* @param[in] copy_points boolean value stating whether the hypervolume computation may use original set
    	*/
    	void set_copy_points(const bool copy_points)
    	{
    		m_copy_points = copy_points;
    	}

    	/// Getter for 'copy_points' flag
    	bool get_copy_points() const
    	{
    		return m_copy_points;
    	}

    	/// Setter for the 'verify' flag
    	/**
    	* Turns off the verification phase.
    	* By default, the hypervolume object verifies whether certain characteristics of the point set hold, such as valid dimension sizes or a reference point that suits the minimisation.
    	* In order to optimize the computation when the rules above are certain, we can turn off that phase.
    	*
    	* This may result in unexpected behaviour when used incorrectly (e.g. requesting the computation of empty set of points)
    	*
    	* @param[in] verify boolean value stating whether the hypervolume computation is to be executed without verification
    	*/
    	void set_verify(const bool verify)
    	{
    		m_verify = verify;
    	}

    	/// Getter for the 'verify' flag
    	bool get_verify() const
    	{
    		return m_verify;
    	}


    	/// Get expected number of operations
    	/**
    	* Returns the expected average amount of elementary operations for given front size (n) and dimension size (d).
    	* This method is used by the approximated algorithms that fall back to exact computation.
    	*
    	* @param[in] n size of the front
    	* @param[in] d dimension size
    	*
    	* @return expected number of operations for given n and d
    	*/
    	static double get_expected_operations(const unsigned int n, const unsigned int d)
    	{
    		if (d <= 3) {
    			return d * n * log(n);  // hv3d
    		}
    		else if (d == 4) {
    			return 4.0 * n * n;  // hv4d
    		}
    		else {
    			return 0.0005 * d * pow(n, d * 0.5);  // exponential complexity
    		}
    	}

    	/// Calculate the default reference point
    	/**
    	* Calculates a mock refpoint by taking the maximum in each dimension over all points saved in the hypervolume object.
    	* The result is a point that is necessarily dominated by all other points, frequently used for hypervolume computations.
    	*
    	* @param[in] offest	value that is to be added to each objective to assure strict domination
    	*
    	* @return reference point
    	*/
    	vector_double get_refpoint(const double offset = 0.0) const {
    		// Corner case
    		if (m_points.size() == 0u) {
    			return {};
    		}

    		auto fdim = m_points[0].size();
    		vector_double ref_point(m_points[0].begin(), m_points[0].end());

    		for (decltype(fdim) f_idx = 0u; f_idx < fdim; ++f_idx) {
    			for (std::vector<vector_double>::size_type idx = 1; idx < m_points.size(); ++idx) {
    				ref_point[f_idx] = std::max(ref_point[f_idx], m_points[idx][f_idx]);
    			}
    		}

    		for (auto &c : ref_point) {
    			c += offset;
    		}

    		return ref_point;
    	}

    	/// Get points
    	/**
    	* Will return a vector containing the points as they were set up during construction of the hypervolume object.
    	*
    	* @return const reference to the vector containing the fitness_vectors representing the points in the hyperspace.
    	*/
    	const std::vector<vector_double> &get_points() const
    	{
    		return m_points;
    	}


    	/// Choose the best hypervolume algorithm for given task
    	/**
    	* Returns the best method for given hypervolume computation problem.
    	* As of yet, only the dimension size is taken into account.
    	*/
    	std::shared_ptr<hv_algorithm> get_best_compute(const vector_double &r_point) const;

    	/// Compute hypervolume
    	/**
    	* Computes hypervolume for given reference point.
    	* This method chooses the hv_algorithm dynamically.
    	*
    	* @param[in] r_point vector describing the reference point
    	*
    	* @return value representing the hypervolume
    	*/
    	double compute(const vector_double &r_point) const
    	{
    		return compute(r_point, get_best_compute(r_point));
    	}


    	/// Compute hypervolume
    	/**
    	* Computes hypervolume for given reference point, using given algorithm object.
    	*
    	* @param[in] r_point fitness vector describing the reference point
    	* @param[in] hv_algorithm instance of the algorithm object used for the computation
    	*
    	* @return value representing the hypervolume
    	*/
    	double compute(const vector_double &r_point, std::shared_ptr<hv_algorithm> hv_algo) const
    	{
    		if (m_verify) {
    			verify_before_compute(r_point, hv_algo);
    		}

    		// copy the initial set of points, as the algorithm may alter its contents
    		if (m_copy_points) {
    			std::vector<vector_double> points_cpy(m_points.begin(), m_points.end());
    			return hv_algo->compute(points_cpy, r_point);
    		}
    		else {
    			return hv_algo->compute(const_cast<std::vector<vector_double> &>(m_points), r_point);
    		}
    	}



    private:
    	std::vector<vector_double> m_points;
    	bool m_copy_points;
    	bool m_verify;

    	/// Verify after construct method
    	/**
    	* Verifies whether basic requirements are met for the initial set of points.
    	*
    	* @throws invalid_argument if point size is empty or when the dimensions among the points differ
    	*/
    	void verify_after_construct() const
    	{
    		if (m_points.size() == 0) {
    			pagmo_throw(std::invalid_argument, "Point set cannot be empty.");
    		}
    		auto f_dim = m_points[0].size();
    		if (f_dim <= 1) {
    			pagmo_throw(std::invalid_argument, "Points of dimension > 1 required.");
    		}
    		for (const auto &v : m_points) {
    			if (v.size() != f_dim) {
    				pagmo_throw(std::invalid_argument, "All point set dimensions must be equal.");
    			}
    		}
    	}

    	/// Verify before compute method
    	/**
    	* Verifies whether reference point and the hypervolume method meet certain criteria.
    	*
    	* @param[in] r_point vector describing the reference point
    	*
    	* @throws value_error if reference point's and point set dimension do not agree
    	*/
    	void verify_before_compute(const vector_double &r_point, std::shared_ptr<hv_algorithm> hv_algo) const
    	{
    		if (m_points[0].size() != r_point.size()) {
    			pagmo_throw(std::invalid_argument, "Point set dimensions and reference point dimension must be equal.");
    		}
    		hv_algo->verify_before_compute(m_points, r_point);
    	}



    	/*hv_algorithm::base_ptr hypervolume::get_best_exclusive(const unsigned int p_idx, const fitness_vector &r_point) const
    	{
    		(void)p_idx;
    		// Exclusive contribution and compute method share the same "best" set of algorithms.
    		return get_best_compute(r_point);
    	}

    	hv_algorithm::base_ptr hypervolume::get_best_contributions(const fitness_vector &r_point) const
    	{
    		unsigned int fdim = r_point.size();
    		if (fdim == 2) {
    			return hv_algorithm::hv2d().clone();
    		}
    		else if (fdim == 3) {
    			return hv_algorithm::hv3d().clone();
    		}
    		else {
    			return hv_algorithm::wfg().clone();
    		}
    	}*/


    };

/// hv2d hypervolume algorithm class
/**
 * This is the class containing the implementation of the hypervolume algorithm for the 2-dimensional fronts.
 * This method achieves the lower bound of n*log(n) time by sorting the initial set of points and then computing the partial areas linearly.
 *
 * @author Krzysztof Nowak (kn@kiryx.net)
 */
class hv2d : public hv_algorithm
{
public:
	/// Constructor
	hv2d(const bool initial_sorting = true) : m_initial_sorting(initial_sorting) { }

	/// Compute hypervolume method.
	/**
	* This method should be used both as a solution to 2-dimensional cases, and as a general termination method for algorithms that reduce n-dimensional problem to 2D.
	*
	* Computational complexity: n*log(n)
	*
	* @param[in] points vector of points containing the 2-dimensional points for which we compute the hypervolume
	* @param[in] r_point reference point for the points
	*
	* @return hypervolume
	*/
	double compute(std::vector<vector_double> &points, const vector_double &r_point) const
	{
		if (points.size() == 0) {
			return 0.0;
		}
		else if (points.size() == 1) {
			return hv_algorithm::volume_between(points[0], r_point);
		}

		if (m_initial_sorting) {
			sort(points.begin(), points.end(), vector_double_cmp(1, '<'));
		}

		double hypervolume = 0.0;

		// width of the sweeping line
		double w = r_point[0] - points[0][0];
		for (unsigned int idx = 0; idx < points.size() - 1; ++idx) {
			hypervolume += (points[idx + 1][1] - points[idx][1]) * w;
			w = std::max(w, r_point[0] - points[idx + 1][0]);
		}
		hypervolume += (r_point[1] - points[points.size() - 1][1]) * w;

		return hypervolume;
	}


	/// Compute hypervolume method.
	/**
	* This method should be used both as a solution to 2-dimensional cases, and as a general termination method for algorithms that reduce n-dimensional problem to 2d.
	* This method is overloaded to work with arrays of double, in order to provide other algorithms that internally work with arrays (such as hv_algorithm::wfg) with an efficient computation.
	*
	* Computational complexity: n*log(n)
	*
	* @param[in] points array of 2-dimensional points
	* @param[in] n_points number of points
	* @param[in] r_point 2-dimensional reference point for the points
	*
	* @return hypervolume
	*/
	double compute(double** points, unsigned int n_points, double* r_point) const
	{
		if (n_points == 0) {
			return 0.0;
		}
		else if (n_points == 1) {
			return volume_between(points[0], r_point, 2);
		}

		if (m_initial_sorting) {
			std::sort(points, points + n_points, hv2d::cmp_double_2d);
		}

		double hypervolume = 0.0;

		// width of the sweeping line
		double w = r_point[0] - points[0][0];
		for (unsigned int idx = 0; idx < n_points - 1; ++idx) {
			hypervolume += (points[idx + 1][1] - points[idx][1]) * w;
			w = std::max(w, r_point[0] - points[idx + 1][0]);
		}
		hypervolume += (r_point[1] - points[n_points - 1][1]) * w;

		return hypervolume;
	}


	/// Contributions method
	/**
	* Computes the contributions of each point by invoking the HV3D algorithm with mock third dimension.
	*
	* @param[in] points vector of points containing the 2-dimensional points for which we compute the hypervolume
	* @param[in] r_point reference point for the points
	* @return vector of exclusive contributions by every point
	*/
	std::vector<double> contributions(std::vector<vector_double> &points, const vector_double &r_point) const;


	/// Clone method.
	std::shared_ptr<hv_algorithm> clone() const
	{
		return std::shared_ptr<hv_algorithm>(new hv2d(*this));
	}


	/// Verify input method.
	/**
	* Verifies whether the requested data suits the hv2d algorithm.
	*
	* @param[in] points vector of points containing the d dimensional points for which we compute the hypervolume
	* @param[in] r_point reference point for the vector of points
	*
	* @throws value_error when trying to compute the hypervolume for the dimension other than 3 or non-maximal reference point
	*/
	void verify_before_compute(const std::vector<vector_double> &points, const vector_double &r_point) const
	{
		if (r_point.size() != 2) {
			pagmo_throw(std::invalid_argument, "Algorithm hv2d works only for 2-dimensional cases.");
		}

		hv_algorithm::assert_minimisation(points, r_point);
	}

	/// Algorithm name
	std::string get_name() const
	{
		return "hv2d algorithm";
	}


private:
	// Flag stating whether the points should be sorted in the first step of the algorithm.
	const bool m_initial_sorting;


	/// Comparison function for sorting of pairs (point, index)
	/**
	* Required for hv2d::extreme_contributor method for keeping track of the original indices when sorting.
	*/
	static bool point_pairs_cmp(const std::pair<vector_double, unsigned int> &a, const std::pair<vector_double, unsigned int> &b)
	{
		return a.first[1] > b.first[1];
	}


	/// Comparison function for arrays of double.
	/**
	* Required by the hv2d::compute method for the sorting of arrays of double*.
	*/
	static bool cmp_double_2d(double* a, double* b)
	{
		return a[1] < b[1];
	}
};

// WFG hypervolume algorithm
/**
 * This is the class containing the implementation of the WFG algorithm for the computation of hypervolume indicator.
 *
 * @see "While, Lyndon, Lucas Bradstreet, and Luigi Barone. "A fast way of calculating exact hypervolumes." Evolutionary Computation, IEEE Transactions on 16.1 (2012): 86-95."
 * @see "Lyndon While and Lucas Bradstreet. Applying the WFG Algorithm To Calculate Incremental Hypervolumes. 2012 IEEE Congress on Evolutionary Computation. CEC 2012, pages 489-496. IEEE, June 2012."
 *
 * @author Krzysztof Nowak (kn@linux.com)
 * @author Marcus M�rtens (mmarcusx@gmail.com)
 */
class hvwfg : public hv_algorithm
{
public:
	/// Constructor
	hvwfg(const unsigned int stop_dimension = 2) : hv_algorithm(), m_current_slice(0), m_stop_dimension(stop_dimension)
	{
		if (stop_dimension < 2) {
			pagmo_throw(std::invalid_argument, "Stop dimension for WFG must be greater than or equal to 2");
		}
	}

	/// Compute hypervolume
	/**
	* Computes the hypervolume using the WFG algorithm.
	*
	* @param[in] points vector of points containing the D-dimensional points for which we compute the hypervolume
	* @param[in] r_point reference point for the points
	*
	* @return hypervolume.
	*/
	double compute(std::vector<vector_double> &points, const vector_double &r_point) const
	{
		allocate_wfg_members(points, r_point);
		double hv = compute_hv(1);
		free_wfg_members();
		return hv;
	}

	/// Contributions method
	/**
	* This method employs a slightly modified version of the original WFG algorithm to suit the computation of the exclusive contributions.
	* It differs from the IWFG algorithm (referenced below), as we do not use the priority-queueing mechanism, but compute every exclusive contribution instead.
	* This may suggest that the algorithm for the extreme contributor itself reduces to the 'naive' approach. It is not the case however,
	* as we utilize the benefits of the 'limitset', before we begin the recursion.
	* This simplifies the sub problems for each exclusive computation right away, which makes the whole algorithm much faster, and in many cases only slower than regular WFG algorithm by a constant factor.
	*
	* @see "Lyndon While and Lucas Bradstreet. Applying the WFG Algorithm To Calculate Incremental Hypervolumes. 2012 IEEE Congress on Evolutionary Computation. CEC 2012, pages 489-496. IEEE, June 2012."
	*
	* @param[in] points vector of points containing the D-dimensional points for which we compute the hypervolume
	* @param[in] r_point reference point for the points
	*/
	std::vector<double> contributions(std::vector<vector_double> &points, const vector_double &r_point) const
	{
		std::vector<double> c;
		c.reserve(points.size());

		// Allocate the same members as for 'compute' method
		allocate_wfg_members(points, r_point);

		// Prepare the memory for first front
		double** fr = new double*[m_max_points];
		for (unsigned int i = 0; i < m_max_points; ++i) {
			fr[i] = new double[m_current_slice];
		}
		m_frames[m_n_frames] = fr;
		m_frames_size[m_n_frames] = 0;
		++m_n_frames;

		for (unsigned int p_idx = 0; p_idx < m_max_points; ++p_idx) {
			limitset(0, p_idx, 1);
			c.push_back(exclusive_hv(p_idx, 1));
		}

		// Free the contributions and the remaining WFG members
		free_wfg_members();

		return c;
	}

	/// Verify before compute method
	/**
	* Verifies whether given algorithm suits the requested data.
	*
	* @param[in] points vector of points containing the D-dimensional points for which we compute the hypervolume
	* @param[in] r_point reference point for the vector of points
	*
	* @throws value_error when trying to compute the hypervolume for the non-maximal reference point
	*/
	void verify_before_compute(const std::vector<vector_double> &points, const vector_double &r_point) const
	{
		hv_algorithm::assert_minimisation(points, r_point);
	}

	/// Clone method.
	std::shared_ptr<hv_algorithm> clone() const
	{
		return std::shared_ptr<hv_algorithm>(new hvwfg(*this));
	}

	/// Algorithm name
	std::string get_name() const
	{
		return "WFG algorithm";
	}
private:
	/// Limit the set of points to point at p_idx
	void limitset(const unsigned int begin_idx, const unsigned int p_idx, const unsigned int rec_level) const
	{
		double **points = m_frames[rec_level - 1];
		unsigned int n_points = m_frames_size[rec_level - 1];

		int no_points = 0;

		double* p = points[p_idx];
		double** frame = m_frames[rec_level];

		for (unsigned int idx = begin_idx; idx < n_points; ++idx) {
			if (idx == p_idx) {
				continue;
			}

			for (vector_double::size_type f_idx = 0; f_idx < m_current_slice; ++f_idx) {
				frame[no_points][f_idx] = std::max(points[idx][f_idx], p[f_idx]);
			}

			std::vector<int> cmp_results;
			cmp_results.resize(no_points);
			double* s = frame[no_points];

			bool keep_s = true;

			// Check whether any point is dominating the point 's'.
			for (int q_idx = 0; q_idx < no_points; ++q_idx) {
				cmp_results[q_idx] = hv_algorithm::dom_cmp(s, frame[q_idx], m_current_slice);
				if (cmp_results[q_idx] == hv_algorithm::DOM_CMP_B_DOMINATES_A) {
					keep_s = false;
					break;
				}
			}

			// If neither is, remove points dominated by 's' (we store that during the first loop).
			if (keep_s) {
				int prev = 0;
				int next = 0;
				while (next < no_points) {
					if (cmp_results[next] != hv_algorithm::DOM_CMP_A_DOMINATES_B && cmp_results[next] != hv_algorithm::DOM_CMP_A_B_EQUAL) {
						if (prev < next) {
							for (unsigned int d_idx = 0; d_idx < m_current_slice; ++d_idx) {
								frame[prev][d_idx] = frame[next][d_idx];
							}
						}
						++prev;
					}
					++next;
				}
				// Append 's' at the end, if prev==next it's not necessary as it's already there.
				if (prev < next) {
					for (unsigned int d_idx = 0; d_idx < m_current_slice; ++d_idx) {
						frame[prev][d_idx] = s[d_idx];
					}
				}
				no_points = prev + 1;
			}
		}

		m_frames_size[rec_level] = no_points;
	}


	/// Compute the exclusive hypervolume of point at p_idx
	double exclusive_hv(const unsigned int p_idx, const unsigned int rec_level) const
	{
		//double H = hv_algorithm::volume_between(points[p_idx], m_refpoint, m_current_slice);
		double H = hv_algorithm::volume_between(m_frames[rec_level - 1][p_idx], m_refpoint, m_current_slice);

		if (m_frames_size[rec_level] == 1) {
			H -= hv_algorithm::volume_between(m_frames[rec_level][0], m_refpoint, m_current_slice);
		}
		else if (m_frames_size[rec_level] > 1) {
			H -= compute_hv(rec_level + 1);
		}

		return H;
	}


	/// Compute the hypervolume recursively
	double compute_hv(const unsigned int rec_level) const
	{
		double **points = m_frames[rec_level - 1];
		unsigned int n_points = m_frames_size[rec_level - 1];

		// Simple inclusion-exclusion for one and two points
		if (n_points == 1) {
			return hv_algorithm::volume_between(points[0], m_refpoint, m_current_slice);
		}
		else if (n_points == 2) {
			double hv = hv_algorithm::volume_between(points[0], m_refpoint, m_current_slice)
				+ hv_algorithm::volume_between(points[1], m_refpoint, m_current_slice);
			double isect = 1.0;
			for (unsigned int i = 0; i<m_current_slice; ++i) {
				isect *= (m_refpoint[i] - std::max(points[0][i], points[1][i]));
			}
			return hv - isect;
		}

		// If already sliced to dimension at which we use another algorithm.
		if (m_current_slice == m_stop_dimension) {

			if (m_stop_dimension == 2) {
				// Use a very efficient version of hv2d
				return hv2d().compute(points, n_points, m_refpoint);
			}
			else {
				// Let hypervolume object pick the best method otherwise.
				std::vector<vector_double> points_cpy;
				points_cpy.reserve(n_points);
				for (unsigned int i = 0; i < n_points; ++i) {
					points_cpy.push_back(vector_double(points[i], points[i] + m_current_slice));
				}
				vector_double r_cpy(m_refpoint, m_refpoint + m_current_slice);

				hypervolume hv = hypervolume(points_cpy, false);
				hv.set_copy_points(false);
				return hv.compute(r_cpy);
			}
		}
		else {
			// Otherwise, sort the points in preparation for the next recursive step
			// Bind the object under "this" pointer to the cmp_points method so it can be used as a valid comparator function for std::sort
			// We need that in order for the cmp_points to have acces to the m_current_slice member variable.
			//std::sort(points, points + n_points, boost::bind(&wfg::cmp_points, this, _1, _2));
			std::sort(points, points + n_points, [this](auto a, auto b) {return this->cmp_points(a, b); });
		}

		double H = 0.0;
		--m_current_slice;

		if (rec_level >= m_n_frames) {
			double** fr = new double*[m_max_points];
			for (unsigned int i = 0; i < m_max_points; ++i) {
				fr[i] = new double[m_current_slice];
			}
			m_frames[m_n_frames] = fr;
			m_frames_size[m_n_frames] = 0;
			++m_n_frames;
		}

		for (unsigned int p_idx = 0; p_idx < n_points; ++p_idx) {
			limitset(p_idx + 1, p_idx, rec_level);

			H += fabs((points[p_idx][m_current_slice] - m_refpoint[m_current_slice]) * exclusive_hv(p_idx, rec_level));
		}
		++m_current_slice;
		return H;
	}


	/// Comparator function for sorting
	/**
	* Comparison function for WFG. Can't be static in order to have access to member variable m_current_slice.
	*/
	bool cmp_points(double* a, double* b) const
	{
		for (int i = m_current_slice - 1; i >= 0; --i) {
			if (a[i] > b[i]) {
				return true;
			}
			else if (a[i] < b[i]) {
				return false;
			}
		}
		return false;
	}


	/// Allocate the memory for the 'compute' method
	void allocate_wfg_members(std::vector<vector_double> &points, const vector_double &r_point) const
	{
		m_max_points = points.size();
		m_max_dim = r_point.size();

		m_refpoint = new double[m_max_dim];
		for (unsigned int d_idx = 0; d_idx < m_max_dim; ++d_idx) {
			m_refpoint[d_idx] = r_point[d_idx];
		}

		// Reserve the space beforehand for each level or recursion.
		// WFG with slicing feature will not go recursively deeper than the dimension size.
		m_frames = new double**[m_max_dim];
		m_frames_size = new unsigned int[m_max_dim];

		// Copy the initial set into the frame at index 0.
		double** fr = new double*[m_max_points];
		for (unsigned int p_idx = 0; p_idx < m_max_points; ++p_idx) {
			fr[p_idx] = new double[m_max_dim];
			for (unsigned int d_idx = 0; d_idx < m_max_dim; ++d_idx) {
				fr[p_idx][d_idx] = points[p_idx][d_idx];
			}
		}
		m_frames[0] = fr;
		m_frames_size[0] = m_max_points;
		m_n_frames = 1;

		// Variable holding the current "depth" of dimension slicing. We progress by slicing dimensions from the end.
		m_current_slice = m_max_dim;
	}


	/// Free the previously allocated memory
	void free_wfg_members() const
	{
		// Free the memory.
		delete[] m_refpoint;

		for (unsigned int fr_idx = 0; fr_idx < m_n_frames; ++fr_idx) {
			for (unsigned int p_idx = 0; p_idx < m_max_points; ++p_idx) {
				delete[] m_frames[fr_idx][p_idx];
			}
			delete[] m_frames[fr_idx];
		}
		delete[] m_frames;
		delete[] m_frames_size;
	}

	/**
	 * 'compute' and 'extreme_contributor' method variables section.
	 *
	 * Variables below (especially the pointers m_frames, m_frames_size and m_refpoint) are initialized
	 * at the beginning of the 'compute' and 'extreme_contributor' methods, and freed afterwards.
	 * The state of the variables is irrelevant outside the scope of the these methods.
	 */

	// Current slice depth
	mutable unsigned int m_current_slice;

	// Array of point sets for each recursive level.
	mutable double*** m_frames;

	// Maintains the number of points at given recursion level.
	mutable unsigned int* m_frames_size;

	// Keeps track of currently allocated number of frames.
	mutable unsigned int m_n_frames;

	// Copy of the reference point
	mutable double* m_refpoint;

	// Size of the original front
	mutable unsigned int m_max_points;

	// Size of the dimension
	mutable unsigned int m_max_dim;
	/**
	 * End of 'compute' method variables section.
	 */

	// Dimension at which WFG stops the slicing
	const unsigned int m_stop_dimension;
};

/// hv3d hypervolume algorithm class
/**
 * This class contains the implementation of efficient algorithms for the hypervolume computation in 3-dimensions.
 *
 * 'compute' method relies on the efficient algorithm as it was presented by Nicola Beume et al.
 * 'least[greatest]_contributor' methods rely on the HyCon3D algorithm by Emmerich and Fonseca.
 *
 * @see "On the Complexity of Computing the Hypervolume Indicator", Nicola Beume, Carlos M. Fonseca, Manuel Lopez-Ibanez, Luis Paquete, Jan Vahrenhold. IEEE TRANSACTIONS ON EVOLUTIONARY COMPUTATION, VOL. 13, NO. 5, OCTOBER 2009
 * @see "Computing hypervolume contribution in low dimensions: asymptotically optimal algorithm and complexity results", Michael T. M. Emmerich, Carlos M. Fonseca
 *
 * @author Krzysztof Nowak (kn@linux.com)
 * @author Marcus M�rtens (mmarcusx@gmail.com)
 */
class hv3d : public hv_algorithm
{
public:
	/**
	* Constructor of the algorithm object.
	* In the very first step, algorithm requires the inital set of points to be sorted ASCENDING in the third dimension.
	* If the input is already sorted, user can skip this step using "initial_sorting = false" option, saving some extra time.
	*
	* @param[in] initial_sorting when set to true (default), algorithm will sort the points ascending by third dimension
	*/
	hv3d(const bool initial_sorting = true) : m_initial_sorting(initial_sorting) { }

	/// Compute hypervolume
	/**
	* This method should be used both as a solution to 3D cases, and as a general termination method for algorithms that reduce D-dimensional problem to 3-dimensional one.
	*
	* This is the implementation of the algorithm for computing hypervolume as it was presented by Nicola Beume et al.
	* The implementation uses std::multiset (which is based on red-black tree data structure) as a container for the sweeping front.
	* Original implementation by Beume et. al uses AVL-tree.
	* The difference is insiginificant as the important characteristics (maintaining order when traversing, self-balancing) of both structures and the asymptotic times (O(log n) updates) are guaranteed.
	* Computational complexity: O(n*log(n))
	*
	* @param[in] points vector of points containing the 3-dimensional points for which we compute the hypervolume
	* @param[in] r_point reference point for the points
	*
	* @return hypervolume.
	*/
	double compute(std::vector<vector_double> &points, const vector_double &r_point) const
	{
		if (m_initial_sorting) {
			sort(points.begin(), points.end(), vector_double_cmp(2, '<'));
		}
		double V = 0.0; // hypervolume
		double A = 0.0; // area of the sweeping plane
		std::multiset<vector_double, vector_double_cmp> T(vector_double_cmp(0, '>'));

		// sentinel points (r_point[0], -INF, r_point[2]) and (-INF, r_point[1], r_point[2])
		const double INF = std::numeric_limits<double>::max();
		vector_double sA(r_point.begin(), r_point.end()); sA[1] = -INF;
		vector_double sB(r_point.begin(), r_point.end()); sB[0] = -INF;

		T.insert(sA);
		T.insert(sB);
		double z3 = points[0][2];
		T.insert(points[0]);
		A = fabs((points[0][0] - r_point[0]) * (points[0][1] - r_point[1]));

		std::multiset<vector_double>::iterator p;
		std::multiset<vector_double>::iterator q;
		for (std::vector<vector_double>::size_type idx = 1; idx < points.size(); ++idx) {
			p = T.insert(points[idx]);
			q = (p);
			++q; //setup q to be a successor of p
			if ((*q)[1] <= (*p)[1]) { // current point is dominated
				T.erase(p); // disregard the point from further calculation
			}
			else {
				V += A * fabs(z3 - (*p)[2]);
				z3 = (*p)[2];
				std::multiset<vector_double>::reverse_iterator rev_it(q);
				++rev_it;

				std::multiset<vector_double>::reverse_iterator erase_begin(rev_it);
				std::multiset<vector_double>::reverse_iterator rev_it_pred;
				while ((*rev_it)[1] >= (*p)[1]) {
					rev_it_pred = rev_it;
					++rev_it_pred;
					A -= fabs(((*rev_it)[0] - (*rev_it_pred)[0])*((*rev_it)[1] - (*q)[1]));
					++rev_it;
				}
				A += fabs(((*p)[0] - (*(rev_it))[0])*((*p)[1] - (*q)[1]));
				T.erase(rev_it.base(), erase_begin.base());
			}
		}
		V += A * fabs(z3 - r_point[2]);

		return V;
	}

	/// Contributions method
	/*
	* This method is the implementation of the HyCon3D algorithm.
	* This algorithm computes the exclusive contribution to the hypervolume by every point, using an efficient HyCon3D algorithm by Emmerich and Fonseca.
	*
	* @see "Computing hypervolume contribution in low dimensions: asymptotically optimal algorithm and complexity results", Michael T. M. Emmerich, Carlos M. Fonseca
	*
	* @param[in] points vector of points containing the 3-dimensional points for which we compute the hypervolume
	* @param[in] r_point reference point for the points
	* @return vector of exclusive contributions by every point
	*/
	std::vector<double> contributions(std::vector<vector_double> &points, const vector_double &r_point) const
	{
		// Make a copy of the original set of points
		std::vector<vector_double> p(points.begin(), points.end());

		std::vector<std::pair<vector_double, unsigned int> > point_pairs;
		point_pairs.reserve(p.size());
		for (unsigned int i = 0; i < p.size(); ++i) {
			point_pairs.push_back(std::make_pair(p[i], i));
		}
		if (m_initial_sorting) {
			sort(point_pairs.begin(), point_pairs.end(), hycon3d_sort_cmp);
		}
		for (unsigned int i = 0; i < p.size(); ++i) {
			p[i] = point_pairs[i].first;
		}

		typedef std::multiset<std::pair<vector_double, int>, hycon3d_tree_cmp > tree_t;

		unsigned int n = p.size();
		const double INF = std::numeric_limits<double>::max();

		// Placeholder value for undefined lower z value.
		const double NaN = INF;

		// Contributions
		std::vector<double> c(n, 0.0);

		// Sentinel points
		vector_double s_x(3, -INF); s_x[0] = r_point[0]; // (r,oo,oo)
		vector_double s_y(3, -INF); s_y[1] = r_point[1]; // (oo,r,oo)
		vector_double s_z(3, -INF); s_z[2] = r_point[2]; // (oo,oo,r)

		p.push_back(s_z); // p[n]
		p.push_back(s_x); // p[n + 1]
		p.push_back(s_y); // p[n + 2]

		tree_t T;
		T.insert(std::make_pair(p[0], 0));
		T.insert(std::make_pair(s_x, n + 1));
		T.insert(std::make_pair(s_y, n + 2));

		// Boxes
		std::vector<std::deque<box3d> > L(n + 3);

		box3d b(r_point[0], r_point[1], NaN, p[0][0], p[0][1], p[0][2]);
		L[0].push_front(b);

		for (unsigned int i = 1; i < n + 1; ++i) {
			std::pair<vector_double, int> pi(p[i], i);

			tree_t::iterator it = T.lower_bound(pi);

			// Point is dominated
			if (p[i][1] >= (*it).first[1]) {
				return hvwfg(2).contributions(points, r_point);
			}

			tree_t::reverse_iterator r_it(it);

			std::vector<int> d;

			while ((*r_it).first[1] > p[i][1]) {
				d.push_back((*r_it).second);
				++r_it;
			}

			int r = (*it).second;
			int t = (*r_it).second;

			T.erase(r_it.base(), it);

			// Process right neighbor region, region R
			while (!L[r].empty()) {
				box3d& b = L[r].front();
				if (b.ux >= p[i][0]) {
					b.lz = p[i][2];
					c[r] += box_volume(b);
					L[r].pop_front();
				}
				else if (b.lx > p[i][0]) {
					b.lz = p[i][2];
					c[r] += box_volume(b);
					b.lx = p[i][0];
					b.uz = p[i][2];
					b.lz = NaN;
					break;
				}
				else {
					break;
				}
			}

			// Process dominated points, region M
			double xleft = p[t][0];
			std::vector<int>::reverse_iterator r_it_idx = d.rbegin();
			std::vector<int>::reverse_iterator r_it_idx_e = d.rend();
			for (; r_it_idx != r_it_idx_e; ++r_it_idx) {
				int jdom = *r_it_idx;
				while (!L[jdom].empty()) {
					box3d& b = L[jdom].front();
					b.lz = p[i][2];
					c[jdom] += box_volume(b);
					L[jdom].pop_front();
				}
				L[i].push_back(box3d(xleft, p[jdom][1], NaN, p[jdom][0], p[i][1], p[i][2]));
				xleft = p[jdom][0];
			}
			L[i].push_back(box3d(xleft, p[r][1], NaN, p[i][0], p[i][1], p[i][2]));
			xleft = p[t][0];

			// Process left neighbor region, region L
			while (!L[t].empty()) {
				box3d &b = L[t].back();
				if (b.ly > p[i][1]) {
					b.lz = p[i][2];
					c[t] += box_volume(b);
					xleft = b.lx;
					L[t].pop_back();
				}
				else {
					break;
				}
			}
			if (xleft > p[t][0]) {
				L[t].push_back(box3d(xleft, p[i][1], NaN, p[t][0], p[t][1], p[i][2]));
			}
			T.insert(std::make_pair(p[i], i));
		}

		// Fix the indices
		std::vector<double> contribs(n, 0.0);
		for (unsigned int i = 0; i < c.size(); ++i) {
			contribs[point_pairs[i].second] = c[i];
		}
		return contribs;
	}


	/// Verify before compute
	/**
	* Verifies whether given algorithm suits the requested data.
	*
	* @param[in] points vector of points containing the d dimensional points for which we compute the hypervolume
	* @param[in] r_point reference point for the vector of points
	*
	* @throws value_error when trying to compute the hypervolume for the dimension other than 3 or non-maximal reference point
	*/
	void verify_before_compute(const std::vector<vector_double> &points, const vector_double &r_point) const
	{
		if (r_point.size() != 3) {
			pagmo_throw(std::invalid_argument, "Algorithm hv3d works only for 3-dimensional cases");
		}

		hv_algorithm::assert_minimisation(points, r_point);
	}

	/// Clone method.
	std::shared_ptr<hv_algorithm> clone() const
	{
		return std::shared_ptr<hv_algorithm>(new hv3d(*this));
	}

	/// Algorithm name
	std::string get_name() const
	{
		return "hv3d algorithm";
	}

private:
	// flag stating whether the points should be sorted in the first step of the algorithm
	const bool m_initial_sorting;

	struct box3d
	{
		box3d(double _lx, double _ly, double _lz, double _ux, double _uy, double _uz)
			: lx(_lx), ly(_ly), lz(_lz), ux(_ux), uy(_uy), uz(_uz) { }
		double lx;
		double ly;
		double lz;
		double ux;
		double uy;
		double uz;
	};

	struct hycon3d_tree_cmp
	{
	//	bool operator()(const std::pair<vector_double, int> &, const std::pair<vector_double, int> &);
		bool operator()(const std::pair<vector_double, int> &a, const std::pair<vector_double, int> &b)
		{
			return a.first[0] > b.first[0];
		}

	};

	/// Comparator method for the hycon3d algorithm's sorting procedure
	static bool hycon3d_sort_cmp(const std::pair<vector_double, unsigned int> &a, const std::pair<vector_double, unsigned int> &b)
	{
		return a.first[2] < b.first[2];
	}

	/// Box volume method
	/**
	* Returns the volume of the box3d object
	*/
	static double box_volume(const box3d &b)
	{
		return fabs((b.ux - b.lx) * (b.uy - b.ly) * (b.uz - b.lz));
	}

};

inline std::vector<double> hv2d::contributions(std::vector<vector_double> &points, const vector_double &r_point) const
{
    std::vector<vector_double> new_points(points.size(), vector_double(3, 0.0));
    vector_double new_r(r_point);
    new_r.push_back(1.0);

    for (unsigned int i = 0; i < points.size(); ++i) {
        new_points[i][0] = points[i][0];
        new_points[i][1] = points[i][1];
        new_points[i][2] = 0.0;
    }
    // Set sorting to off since contributions are sorted by third dimension
    return hv3d(false).contributions(new_points, new_r);
}

/// Choose the best hypervolume algorithm for given task
/**
* Returns the best method for given hypervolume computation problem.
* As of yet, only the dimension size is taken into account.
*/
inline std::shared_ptr<hv_algorithm> hypervolume::get_best_compute(const vector_double &r_point) const
{
    unsigned int fdim = r_point.size();
    unsigned int n = m_points.size();
    if (fdim == 2) {
        return hv2d().clone();
    }
    /*else if (fdim == 3) {
    return hv_algorithm::hv3d().clone();
    }
    else if (fdim == 4) {
    return hv_algorithm::hv4d().clone();
    }
    else if (fdim == 5 && n < 80) {
    return hv_algorithm::fpl().clone();
    }
    else {
    return hv_algorithm::wfg().clone();
    }*/
    else {
        pagmo_throw(std::invalid_argument, "Current implementation allows only to compute 2d hypervolumes!");
    }
}

} // namespace pagmo

#endif
