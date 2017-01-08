#include <iostream>
#include <vector>

int main()
{
    static thread_local std::vector<char> n;
    std::cout << n.size() << std::endl;
}
