extern "C"
{
#include <pthread.h>
#include <sched.h>
}

int main()
{
        ::cpu_set_t cpuset;
        ::pthread_t thread;
        int (*fptr)(::pthread_t thread, size_t cpusetsize,
                const ::cpu_set_t *cpuset) = pthread_setaffinity_np;
}
