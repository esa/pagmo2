namespace pagmo
{

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

		auto n = p.size();
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
    auto fdim = r_point.size();
    auto n = m_points.size();
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

}
