#include <memory>
#include <utility>
#include <type_traits>
#include <typeinfo>

#include <iostream>
#include <string>

using namespace std::string_literals;

class algorithm
{
        struct inner_base
        {
            virtual ~inner_base() {}
            virtual inner_base *clone() const = 0;
            virtual void evolve() const = 0;
        };
        template <typename T>
        struct inner: inner_base
        {
            explicit inner(T &&x):m_value(std::move(x)) {}
            explicit inner(const T &x):m_value(x) {}
            virtual inner_base *clone() const override
            {
                return ::new inner<T>(m_value);
            }
            virtual void evolve() const override
            {
                return m_value.evolve();
            }
            T m_value;
        };
    public:
        template <typename T>
        explicit algorithm(T &&x):m_ptr(::new inner<std::decay_t<T>>(std::forward<T>(x))) {}
        algorithm(const algorithm &other):m_ptr(other.m_ptr->clone()) {}
        algorithm(algorithm &&other) = default;
        template <typename T>
        const T &any_cast() const
        {
            auto ptr = dynamic_cast<const inner<T> *>(m_ptr.get());
            if (ptr == nullptr) {
                throw std::bad_cast{};
            }
            return ptr->m_value;
        }
        void evolve() const
        {
            return m_ptr->evolve();
        }
    private:
        std::unique_ptr<inner_base> m_ptr;
};

struct differential_evolution
{
    void evolve() const {};
};

struct sga
{
    void evolve() const {};
};

int main()
{
    algorithm a{differential_evolution{}}, b{sga{}};
    a.evolve();
    b.evolve();
    a.any_cast<differential_evolution>();
    b.any_cast<sga>();
    // This will throw.
    // a.any_cast<sga>();
}

