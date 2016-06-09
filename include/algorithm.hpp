#ifndef PAGMO_ALGORITHM_HPP
#define PAGMO_ALGORITHM_HPP

#include <memory>
#include <type_traits>
#include <typeinfo>
#include <utility>

#include "serialization.hpp"
#include "population.hpp"

#define PAGMO_REGISTER_ALGORITHM(algo) CEREAL_REGISTER_TYPE_WITH_NAME(pagmo::detail::algo_inner<algo>,#algo)

namespace pagmo
{

/// Type has set_verbose
/**
 * This type trait defines a static const boolean
 * \p value flag which is \p true if the expression p.set_verbose(n)
 * is valid and returns void, where p is a non-const instance of \p T and n is a bool
 *
 * For example, if \p T has the following method implemented:
 *
 * @code
 * void set_verbose(unsigned int level)
 * @endcode
 *
 */
template <typename T>
class has_set_verbosity: detail::sfinae_types
{
        template <typename U>
        static auto test0(U &p) -> decltype(p.set_verbosity(std::declval<unsigned int>()));
        static no test0(...);
        static const bool implementation_defined =
            std::is_same<void,decltype(test0(std::declval<T &>()))>::value;
    public:
        /// static const boolean value flag
        static const bool value = implementation_defined;
};

template <typename T>
const bool has_set_verbosity<T>::value;

/// Type has has_set_verbosity()
/**
 * This type trait defines a static const boolean
 * \p value flag which is \p true if \p T has the following
 * method implemented:
 *
 * @code
 * bool has_set_verbosity() const
 * @endcode
 *
 */
template <typename T>
class override_has_set_verbosity: detail::sfinae_types
{
        template <typename U>
        static auto test0(const U &p) -> decltype(p.has_set_verbosity());
        static no test0(...);
        static const bool implementation_defined =
            std::is_same<bool,decltype(test0(std::declval<const T &>()))>::value;
    public:
        /// static const boolean value flag
        static const bool value = implementation_defined;
};

template <typename T>
const bool override_has_set_verbosity<T>::value;

/// Type has evolve
/**
 * This type trait defines a static const boolean
 * \p value flag which is \p true if \p T has the following
 * method implemented:
 *
 * @code
 * population evolve(population pop) const
 * @endcode
 */
template <typename T>
class has_evolve: detail::sfinae_types
{
        template <typename U>
        static auto test0(const U &p) -> decltype(p.evolve(std::declval<const population &>()));
        static no test0(...);
        static const bool implementation_defined =
            std::is_same<population,decltype(test0(std::declval<const T &>()))>::value;
    public:
        /// static const boolean value flag
        static const bool value = implementation_defined;
};

template <typename T>
const bool has_evolve<T>::value;

namespace detail
{

struct algo_inner_base
{
    virtual ~algo_inner_base() {}
    virtual algo_inner_base *clone() const = 0;
    virtual population evolve(const population &pop) const = 0;
    virtual void set_seed(unsigned int) = 0;
    virtual bool has_set_seed() const = 0;
    virtual void set_verbosity(unsigned int) = 0;
    virtual bool has_set_verbosity() const = 0;
    virtual std::string get_name() const = 0;
    virtual std::string get_extra_info() const = 0;

    template <typename Archive>
    void serialize(Archive &) {}
};

template <typename T>
struct algo_inner final: algo_inner_base
{
    // Static checks.
    static_assert(
        std::is_default_constructible<T>::value &&
        std::is_copy_constructible<T>::value &&
        std::is_move_constructible<T>::value &&
        std::is_destructible<T>::value,
        "An algorithm must be default-constructible, copy-constructible, move-constructible and destructible."
    );
    static_assert(has_evolve<T>::value,
        "A user-defined algorithm must provide a method with signature 'population evolve(population) const'. Could not detect one.");
    // We just need the def ctor, delete everything else.
    algo_inner() = default;
    algo_inner(const algo_inner &) = delete;
    algo_inner(algo_inner &&) = delete;
    algo_inner &operator=(const algo_inner &) = delete;
    algo_inner &operator=(algo_inner &&) = delete;
    // Constructors from T (copy and move variants).
    explicit algo_inner(T &&x):m_value(std::move(x)) {}
    explicit algo_inner(const T &x):m_value(x) {}
    // The clone method, used in the copy constructor of algorithm.
    virtual algo_inner_base *clone() const override final
    {
        return ::new algo_inner<T>(m_value);
    }
    // Mandatory methods.
    virtual population evolve(const population& pop) const override final
    {
        return m_value.evolve(pop);
    }
    // Optional methods
    virtual void set_seed(unsigned int seed) override final
    {
        set_seed_impl(m_value, seed);
    }
    virtual bool has_set_seed() const override final
    {
        return has_set_seed_impl(m_value);
    }
    virtual void set_verbosity(unsigned int level) override final
    {
        set_verbosity_impl(m_value, level);
    }
    virtual bool has_set_verbosity() const override final
    {
        return has_set_verbosity_impl(m_value);
    }
    virtual std::string get_name() const override final
    {
        return get_name_impl(m_value);
    }
    virtual std::string get_extra_info() const override final
    {
        return get_extra_info_impl(m_value);
    }
    // Implementation of the optional methods.
    template <typename U, typename std::enable_if<pagmo::has_set_seed<U>::value,int>::type = 0>
    static void set_seed_impl(U &a, unsigned int seed)
    {
        a.set_seed(seed);
    }
    template <typename U, typename std::enable_if<!pagmo::has_set_seed<U>::value,int>::type = 0>
    static void set_seed_impl(U &, unsigned int)
    {
        pagmo_throw(std::logic_error,"The set_seed method has been called but not implemented by the user.\n"
            "A function with prototype 'void set_seed(unsigned int)' was expected in the user defined algorithm.");
    }
    template <typename U, typename std::enable_if<pagmo::has_set_seed<U>::value && override_has_set_seed<U>::value,int>::type = 0>
    static bool has_set_seed_impl(const U &a)
    {
       return a.has_set_seed();
    }
    template <typename U, typename std::enable_if<pagmo::has_set_seed<U>::value && !override_has_set_seed<U>::value,int>::type = 0>
    static bool has_set_seed_impl(const U &)
    {
       return true;
    }
    template <typename U, typename std::enable_if<!pagmo::has_set_seed<U>::value,int>::type = 0>
    static bool has_set_seed_impl(const U &)
    {
       return false;
    }
    template <typename U, typename std::enable_if<pagmo::has_set_verbosity<U>::value,int>::type = 0>
    static void set_verbosity_impl(U &value, unsigned int seed)
    {
        value.set_verbosity(seed);
    }
    template <typename U, typename std::enable_if<!pagmo::has_set_verbosity<U>::value,int>::type = 0>
    static void set_verbosity_impl(U &, unsigned int)
    {
        pagmo_throw(std::logic_error,"The set_verbosity method has been called but not implemented by the user.\n"
            "A function with prototype 'void set_verbosity(unsigned int)' was expected in the user defined algorithm.");
    }
    template <typename U, typename std::enable_if<pagmo::has_set_verbosity<U>::value && override_has_set_verbosity<U>::value,int>::type = 0>
    static bool has_set_verbosity_impl(const U &a)
    {
       return a.has_set_verbosity();
    }
    template <typename U, typename std::enable_if<pagmo::has_set_verbosity<U>::value && !override_has_set_verbosity<U>::value,int>::type = 0>
    static bool has_set_verbosity_impl(const U &)
    {
       return true;
    }
    template <typename U, typename std::enable_if<!pagmo::has_set_verbosity<U>::value,int>::type = 0>
    static bool has_set_verbosity_impl(const U &)
    {
       return false;
    }
    template <typename U, typename std::enable_if<has_name<U>::value,int>::type = 0>
    static std::string get_name_impl(const U &value)
    {
        return value.get_name();
    }
    template <typename U, typename std::enable_if<!has_name<U>::value,int>::type = 0>
    static std::string get_name_impl(const U &)
    {
        return typeid(U).name();
    }
    template <typename U, typename std::enable_if<has_extra_info<U>::value,int>::type = 0>
    static std::string get_extra_info_impl(const U &value)
    {
        return value.get_extra_info();
    }
    template <typename U, typename std::enable_if<!has_extra_info<U>::value,int>::type = 0>
    static std::string get_extra_info_impl(const U &)
    {
        return "";
    }
    // Serialization
    template <typename Archive>
    void serialize(Archive &ar)
    {
        ar(cereal::base_class<algo_inner_base>(this),m_value);
    }
    T m_value;
};

} // end of namespace detail

/// Algorithm class.
/**
 * \image html algorithm.jpg
 *
 * This class represents an optimization algorithm. An algorithm can be
 * stochastic, deterministic, population based, derivative-free, using hessians,
 * using gradients, a meta-heuristic, evolutionary, etc.. Via this class PaGMO offers a common interface to
 * all types of algorithms that can be applied to find solution to a generic matematical
 * programming problem as represented by the pagmo::problem class.
 *
 * To create an instance of a pagmo::algorithm the user is asked to construct it from
 * a separate object of type \p T where, at least, a method must be present having signature
 * equivalent to the following:
 *
 * @code
 * population evolve(const population&) const;
 * @endcode
 *
 * - The signature does not need to have const &.
 * - The return value of \p T::evolve(const population&) is the
 * optimized (*evolved*) pagmo::population.
 *
 * The mandatory method above allow to define a deterministic algorithm.
 * In order to consider more complex cases, the user may implement one or more
 * of the following methods in \p T :
 *   @code
 * void set_seed(unsigned int seed)
 * void set_verbosity(unsigned int level)
 * std::string get_name() const
 * std::string get_extra_info() const
 *   @endcode
 *
 * - \p T::set_seed(unsigned int seed) changes the value of the random seed used in the user implemented evolve
 * method to drive a stochastic optimization.
 * - \p T::set_verbosity(unsigned int level) changes the value of the screen output and logs verbosity as implemented in
 * the user defined algorithm. When not implemented a call to algorithm::set_verbosity(unsigned int level) throws an \p std::logic_error.
 * - \p T::get_name() returns a string containing the algorithm name to be used in output streams.
 * - \p T::get_extra_info() returns a string containing extra human readable information to be used in output streams. For
 * example the algorithm parameters.
 *
 */
class algorithm
{
        // Enable the generic ctor only if T is not an algorithm (after removing
        // const/reference qualifiers).
        template <typename T>
        using generic_ctor_enabler = std::enable_if_t<!std::is_same<algorithm,std::decay_t<T>>::value,int>;
    public:
        /// Constructor from a user algorithm of type \p T
        /**
         * Construct a pagmo::algorithm with from an object of type \p T. In
         * order for the construction to be successfull \p T needs
         * to satisfy the following requests:
         *
         * - \p T must implement the following mandatory method:
         *   @code
         *   population evolve(const population&) const;
         *   @endcode
         *   otherwise construction will result in a compile-time failure
         * - \p T must be default-constructible, copy-constructible, move-constructible and destructible,
         *   otherwise a call to this constructor will result in a compile-time failure
         *
         * @note \p T must be not of type pagmo::algorithm, otherwise this templated constructor is not enabled and the
         * copy constructor will be called instead.
         *
         * @param[in] x The user implemented algorithm
         *
         */
        template <typename T, generic_ctor_enabler<T> = 0>
        explicit algorithm(T &&x):m_ptr(::new detail::algo_inner<std::decay_t<T>>(std::forward<T>(x)))
        {
            // We detect if set_seed is implemented in the algorithm, in which case the algorithm is stochastic
            m_has_set_seed = ptr()->has_set_seed();
            // We detect if set_verbosity is implemented in the algorithm
            m_has_set_verbosity = ptr()->has_set_verbosity();
            // We store at construction the value returned from the user implemented get_name
            m_name = ptr()->get_name();
        }
        /// Copy constructor
        algorithm(const algorithm &other):
            m_ptr(other.m_ptr->clone()),
            m_has_set_seed(other.m_has_set_seed),
            m_has_set_verbosity(other.m_has_set_verbosity),
            m_name(other.m_name)
        {}
        /// Move constructor
        algorithm(algorithm &&other) noexcept :
            m_ptr(std::move(other.m_ptr)),
            m_has_set_seed(std::move(other.m_has_set_seed)),
            m_has_set_verbosity(other.m_has_set_verbosity),
            m_name(std::move(other.m_name))
        {}
        /// Move assignment operator
        algorithm &operator=(algorithm &&other) noexcept
        {
            if (this != &other) {
                m_ptr = std::move(other.m_ptr);
                m_has_set_seed = std::move(other.m_has_set_seed);
                m_has_set_verbosity = other.m_has_set_verbosity;
                m_name = std::move(other.m_name);
            }
            return *this;
        }
        /// Copy assignment operator
        algorithm &operator=(const algorithm &other)
        {
            // Copy ctor + move assignment.
            return *this = algorithm(other);
        }

        /// Extracts the user-defined algorithm
        /**
         * Extracts the original algorithm that was provided by the user, thus
         * granting access to additional resources there implemented.
         *
         * @tparam T The type of the orignal user-defined algorithm
         *
         * @return a const pointer to the user-defined algorithm, or \p nullptr
         * if \p T does not correspond exactly to the original algorithm type used
         * in the constructor.
         */
        template <typename T>
        const T *extract() const
        {
            auto p = dynamic_cast<const detail::algo_inner<T> *>(ptr());
            if (p == nullptr) {
                return nullptr;
            }
            return &(p->m_value);
        }

        /// Checks the user defined algorithm type at run-time
        /**
         * @tparam T The type to be checked
         *
         * @return \p true if the user defined algorithm is \p T, \p false othewise.
         */
        template <typename T>
        bool is() const
        {
            return extract<T>() != nullptr;
        }

        /// Evolve method
        /**
         * Calls the user implemented evolve method. This is where the core of the optimization (*evolution*) is made.
         *
         * @param  pop starting population
         * @return     evolved population
         * @throws unspecified all excpetions thrown by the user implemented evolve method.
         */
        population evolve(const population &pop) const
        {
            return ptr()->evolve(pop);
        }

        /// Sets the seed for stochastic evolution
        /**
         * Sets the seed to be used in the evolve method for all
         * all stochastic variables. This is assuming that the user implemented in his algorithm random generators
         * controlled by the optional set_seed method.
         *
         * @param[in] seed seed
         */
        void set_seed(unsigned int seed)
        {
            ptr()->set_seed(seed);
        }

        /// Check if the user-defined algorithm implements a set_seed method
        /**
         * If the user defined algorithm implements a set_seed method, this
         * will return true, false otherwise. The value returned can
         * also be forced by the user by implementing the additional
         * method
         *
         * @code
         * bool has_set_seed() const
         * @endcode
         *
         * in the user-defined algorithm
         *
         * @return a boolean flag
         *
         */
        bool has_set_seed() const
        {
            return m_has_set_seed;
        }

        /// Alias for algorithm::has_set_seed
        bool is_stochastic() const
        {
            return has_set_seed();
        }

        /// Sets the verbosity of logs and screen output
        /**
         * Sets the level of verbosity. This is assuming that the user implemented in his algorithm a verbosity
         * control controlled by a set_verbosity method.
         *
         * @param[in] level verbosity level
         */
        void set_verbosity(unsigned int level)
        {
            ptr()->set_verbosity(level);
        }

        /// Check if the user-defined algorithm implements a set_verbosity method
        /**
         * If the user defined algorithm implements a set_verbosity method, this
         * will return true, false otherwise. The value returned can
         * also be forced by the user by implementing the additional
         * method
         *
         * @code
         * bool has_set_verbosity() const
         * @endcode
         *
         * in the user-defined algorithm
         *
         * @return a boolean flag
         */
        bool has_set_verbosity() const
        {
            return m_has_set_verbosity;
        }

        /// Algorithm's name.
        /**
         * @return The algorithm's name as returned by the corresponding
         * user-implemented method if present, the C++ mingled class name otherwise.
         */
        std::string get_name() const
        {
            return m_name;
        }

        /// Extra info
        /**
         * @return The algorithm's extra info as returned by the corresponding
         * user-implemented method if present, an empty string otehrwise.
         */
        std::string get_extra_info() const
        {
            return ptr()->get_extra_info();
        }

        /// Streaming operator
        /**
         *
         * @param os input <tt>std::ostream</tt>
         * @param a pagmo::algorithm object to be streamed
         *
         * @return An std::ostream containing a human-readable
         * representation of the algorithm, including the result from
         * the user-defined method extra_info if implemented.
         */
        friend std::ostream &operator<<(std::ostream &os, const algorithm &a)
        {
            os << "Algorithm name: " << a.get_name();
            if (!a.has_set_seed()) {
                stream(os, " [deterministic]");
            }
            else {
                stream(os, " [stochastic]");
            }
            const auto extra_str = a.get_extra_info();
            if (!extra_str.empty()) {
                stream(os, "\nExtra info:\n", extra_str);
            }
            return os;
        }

        /// Serilization
        template <typename Archive>
        void serialize(Archive &ar)
        {
            ar(m_ptr, m_has_set_seed, m_has_set_verbosity, m_name);
        }
    private:
        // Two small helpers to make sure that whenever we require
        // access to the pointer it actually points to something.
        detail::algo_inner_base const *ptr() const
        {
            assert(m_ptr.get() != nullptr);
            return m_ptr.get();
        }
        detail::algo_inner_base *ptr()
        {
            assert(m_ptr.get() != nullptr);
            return m_ptr.get();
        }
    private:
        std::unique_ptr<detail::algo_inner_base> m_ptr;
        // Various algorithm properties determined at construction time
        // from the concrete algorithm. These will be constant for the lifetime
        // of algorithm, but we cannot mark them as such because of serialization.
        // the extra_info string cannot be here as it must reflect the changes from set_seed
        bool m_has_set_seed;
        bool m_has_set_verbosity;
        std::string m_name;
};

}

#endif
