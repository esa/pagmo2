extern "C"
{
#include <pthread.h>
#include <sched.h>
}

int main()
{
	typedef decltype(::pthread_setaffinity_np) fptr_type;
	using set_type = cpu_set_t;
	return 0;
}
