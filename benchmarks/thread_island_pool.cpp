#include <chrono>
#include <string>

#include <pagmo/algorithms/de.hpp>
#include <pagmo/archipelago.hpp>
#include <pagmo/islands/thread_island.hpp>
#include <pagmo/problems/rosenbrock.hpp>

class simple_timer
{
public:
    simple_timer(const char *desc) : m_desc(desc), m_start(std::chrono::high_resolution_clock::now()) {}
    double elapsed() const
    {
        return static_cast<double>(
            std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - m_start)
                .count());
    }
    ~simple_timer()
    {
        std::cout << "Elapsed time for '" + m_desc + "': "
                  << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now()
                                                                      - m_start)
                         .count()
                  << "s\n";
    }

private:
    const std::string m_desc;
    const std::chrono::high_resolution_clock::time_point m_start;
};

using namespace pagmo;

int main()
{
    archipelago archi{1000, thread_island{false}, de{1000}, rosenbrock{5000}, 20};

    {
        simple_timer st{"total runtime"};

        archi.evolve();

        archi.wait_check();
    }
}
