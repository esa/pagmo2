namespace pagmo
{

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


  /// Compute exclusive contribution
  /**
  * Computes exclusive hypervolume for given indivdual.
  *
  * @param[in] p_idx index of the individual for whom we compute the exclusive contribution to the hypervolume
  * @param[in] r_point fitness vector describing the reference point
  * @param[in] hv_algorithm instance of the algorithm object used for the computation
  *
  * @return value representing the hypervolume
  */
  double hypervolume::exclusive(const unsigned int p_idx, const vector_double &r_point, std::shared_ptr<hv_algorithm> hv_algo) const
  {
	  if (m_verify) {
		  verify_before_compute(r_point, hv_algo);
	  }

	  if (p_idx >= m_points.size()) {
		  pagmo_throw(std::invalid_argument, "Index of the individual is out of bounds.");

	  }

	  // copy the initial set of points, as the algorithm may alter its contents
	  if (m_copy_points) {
		  std::vector<vector_double> points_cpy(m_points.begin(), m_points.end());
		  return hv_algo->exclusive(p_idx, points_cpy, r_point);
	  }
	  else {
		  return hv_algo->exclusive(p_idx, const_cast<std::vector<vector_double> &>(m_points), r_point);
	  }
  }

  /// Compute exclusive contribution
  /**
  * Computes exclusive hypervolume for given indivdual.
  * This methods chooses the hv_algorithm dynamically.
  *
  * @param[in] p_idx index of the individual for whom we compute the exclusive contribution to the hypervolume
  * @param[in] r_point fitness vector describing the reference point
  *
  * @return value representing the hypervolume
  */
  /*double hypervolume::exclusive(const unsigned int p_idx, const vector_double &r_point) const
  {
	  return exclusive(p_idx, r_point, get_best_exclusive(p_idx, r_point));
  }*/




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

}
