#ifndef PAGMO_ALGORITHM_HPP
#define PAGMO_ALGORITHM_HPP

#include <memory>
#include <type_traits>
#include <typeinfo>
#include <utility>

#include "serialization.hpp"
#include "population.hpp"

#define PAGMO_REGISTER_ALGORITHM(algo) CEREAL_REGISTER_TYPE_WITH_NAME(pagmo::detail::algo_inner<algo>,#algo);

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
    // The clone method, used in the copy constructor of problem.
    virtual algo_inner_base *clone() const override
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
            "A function with prototype 'void set_seed(unsigned int)' was expected in the user defined problem.");
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
    public:
        template <typename T>
        explicit algorithm(T &&x):m_ptr(::new detail::algo_inner<std::decay_t<T>>(std::forward<T>(x))) {}
        algorithm(const algorithm &other):m_ptr(other.m_ptr->clone()) {}
        algorithm(algorithm &&other) = default;

        template <typename T>
        const T *extract() const
        {
            auto ptr = dynamic_cast<const detail::algo_inner<T> *>(m_ptr.get());
            if (ptr == nullptr) {
                return nullptr;
            }
            return &(ptr->m_value);
        }

        template <typename T>
        bool is() const
        {
            return extract<T>();
        }

        void evolve() const
        {
            return m_ptr->evolve();
        }
        
        template <typename Archive>
        void serialize(Archive &ar)
        { 
            ar(m_ptr); 
        }
    private:
        std::unique_ptr<detail::algo_inner_base> m_ptr;
        bool m_is_stochastic;
        std::string m_name;
        std::string m_extra_info;
};

}

#endif
