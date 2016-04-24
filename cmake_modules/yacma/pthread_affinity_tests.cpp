extern "C"
{
#include <pthread.h>
#include <sched.h>
}

int main()
{
	typedef decltype(::pthread_setaffinity_np) fptr_type;
	typedef cpu_set_t set_type;
	return 0;
}
