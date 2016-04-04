#ifndef PAGMO_ALGORITHM_HPP
#define PAGMO_ALGORITHM_HPP

#include <memory>
#include <type_traits>
#include <typeinfo>
#include <utility>

#include "serialization.hpp"

#define PAGMO_REGISTER_ALGORITHM(algo) CEREAL_REGISTER_TYPE_WITH_NAME(pagmo::detail::algo_inner<algo>,#algo);

namespace pagmo
{

namespace detail
{

struct algo_inner_base
{
    virtual ~algo_inner_base() {}
    virtual algo_inner_base *clone() const = 0;
    virtual void evolve() const = 0;
    template <typename Archive>
    void serialize(Archive &) {}
};

template <typename T>
struct algo_inner: algo_inner_base
{
    algo_inner() = default;
    explicit algo_inner(T &&x):m_value(std::move(x)) {}
    explicit algo_inner(const T &x):m_value(x) {}
    virtual algo_inner_base *clone() const override
    {
        return ::new algo_inner<T>(m_value);
    }
    virtual void evolve() const override
    {
        return m_value.evolve();
    }
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
};

}

#endif
