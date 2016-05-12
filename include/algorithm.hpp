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

namespace detail
{

struct algo_inner_base
{
    virtual ~algo_inner_base() {}
    virtual algo_inner_base *clone() const = 0;
    virtual population evolve(const population &pop) const = 0;
    virtual void set_seed(unsigned int) = 0;
    virtual bool has_set_seed() const = 0;
    virtual void set_verbose(unsigned int) = 0;
    virtual std::string get_name() const = 0;
    virtual std::string get_extra_info() const = 0;

    template <typename Archive>
    void serialize(Archive &) {}
};

template <typename T>
struct algo_inner: algo_inner_base
{
    /// Static checks
    static_assert(has_evolve<T>::value,
        "A user-defined algorithm must provide a method with signature 'population evolve(const population &) const'. Could not detect one. ");
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
    virtual void set_verbose(unsigned int level) override final
    {
        set_verbose_impl(m_value, level);
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
    static void set_seed_impl(U &value, unsigned int seed)
    {
        value.set_seed(seed);
    }
    template <typename U, typename std::enable_if<!pagmo::has_set_seed<U>::value,int>::type = 0>
    static void set_seed_impl(U &, unsigned int)
    {
        pagmo_throw(std::logic_error,"The set_seed method has been called but not implemented by the user.\n"
            "A function with prototype 'void set_seed(unsigned int)' was expected in the user defined algorithm.");
    }
    template <typename U, typename std::enable_if<pagmo::has_set_seed<U>::value && override_has_set_seed<U>::value,int>::type = 0>
    static bool has_set_seed_impl(const U &p)
    {
       return p.has_set_seed();
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
    template <typename U, typename std::enable_if<pagmo::has_set_verbose<U>::value,int>::type = 0>
    static void set_verbose_impl(U &value, unsigned int seed)
    {
        value.set_verbose(seed);
    }
    template <typename U, typename std::enable_if<!pagmo::has_set_verbose<U>::value,int>::type = 0>
    static void set_verbose_impl(U &, unsigned int)
    {}
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

}

class algorithm
{
        // Enable the generic ctor only if T is not an algorithm (after removing
        // const/reference qualifiers).
        template <typename T>
        using generic_ctor_enabler = std::enable_if_t<!std::is_same<algorithm,std::decay_t<T>>::value,int>;
    public:
        /// Constructor
        template <typename T>
        explicit algorithm(T &&x):m_ptr(::new detail::algo_inner<std::decay_t<T>>(std::forward<T>(x)))
        {
            // We detect if set_seed is implemented in the algorithm, in which case the algorithm is stochastic
            m_is_stochastic = ptr()->has_set_seed();
            // We store at construction the values returned from the name and extra info methods.
            m_name = ptr()->get_name();
            m_extra_info = ptr()->get_extra_info();
        }
        /// Copy constructor
        algorithm(const algorithm &other):
            m_ptr(other.m_ptr->clone()), 
            m_is_stochastic(other.m_is_stochastic),
            m_name(other.m_name),
            m_extra_info(other.m_extra_info)
        {}
        /// Move constructor
        algorithm(algorithm &&other) noexcept :
            m_ptr(std::move(other.m_ptr)),
            m_name(other.m_name),
            m_extra_info(other.m_extra_info)
        {}
        /// Move assignment operator
        algorithm &operator=(algorithm &&other) noexcept
        {
            if (this != &other) {
                m_ptr = std::move(other.m_ptr);
                m_name = other.m_name;
                m_extra_info = other.m_extra_info;
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
            auto p = dynamic_cast<const detail::prob_inner<T> *>(ptr());
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
        population evolve(const population &pop) const
        {
            return ptr()->evolve(pop);
        }

        void set_seed(unsigned int seed)
        {
            ptr()->set_seed(seed);
        }

        /// Check if the user-defined algorithm implements a set_seed method
        bool has_set_seed() const
        {
            return is_stochastic();
        }

        /// Check if the user-defined algorithm implements a set_seed method
        bool is_stochastic() const
        {
            return m_is_stochastic;
        }

        /// Get name
        std::string get_name() const
        {
            return m_name;
        }

        /// Extra info
        std::string get_extra_info() const
        {
            return m_extra_info;
        }

        /// Streaming operator
        friend std::ostream &operator<<(std::ostream &os, const algorithm &p)
        {
            os << "Algorithm name: " << p.get_name();
            if (!p.is_stochastic()) {
                stream(os, " [deterministic]");
            }
            const auto extra_str = p.get_extra_info();
            if (!extra_str.empty()) {
                stream(os, "\nAlgorithm's extra info:\n", extra_str);
            }
            return os;
        }

        template <typename Archive>
        void serialize(Archive &ar)
        {
            ar(m_ptr);
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
        bool m_is_stochastic;
        std::string m_name;
        std::string m_extra_info;
};

}

#endif
