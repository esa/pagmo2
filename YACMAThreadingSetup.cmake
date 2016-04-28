# NOTE: it looks like on CMake >= 3.1 there's a thread import target that can be used:
# https://cmake.org/cmake/help/v3.5/module/FindThreads.html
# http://stackoverflow.com/questions/1620918/cmake-and-libpthread

if(YACMAThreadingSetupIncluded)
    return()
endif()

include(CheckCXXCompilerFlag)
include(CheckCXXSymbolExists)

# Initial thread setup.
find_package(Threads REQUIRED)
message(STATUS "Thread library: ${CMAKE_THREAD_LIBS_INIT}")

# Setup variable for threading-specific flags.
set(YACMA_THREADING_CXX_FLAGS "")

# POSIX thread setup. Intended both for UNIX and Windows (the latter when using some sort of
# pthread emulation/wrapper like pthreads-win32).
if(CMAKE_USE_PTHREADS_INIT)
	message(STATUS "POSIX threads detected.")
	# For POSIX threads, we try to see if the compiler accepts the -pthread flag. It is a bit of a kludge,
	# but I do not have any better idea at the moment. The situation is hairy, e.g.,
	# different systems require different GCC flags:
	# http://gcc.gnu.org/onlinedocs/libstdc++/manual/using_concurrency.html
	CHECK_CXX_COMPILER_FLAG(-pthread YACMA_PTHREAD_COMPILER_FLAG)
	# NOTE: we do not enable the -pthread flag on OS X as it is apparently ignored.
	if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin" AND YACMA_PTHREAD_COMPILER_FLAG)
		message(STATUS "Enabling the -pthread compiler flag.")
		# NOTE: according to GCC docs, this sets the flag for both compiler and linker. This should
		# work similarly for clang as well.
		set(YACMA_THREADING_CXX_FLAGS "-pthread")
	endif()
	unset(YACMA_PTHREAD_COMPILER_FLAG)
	set(CMAKE_REQUIRED_LIBRARIES "${CMAKE_THREAD_LIBS_INIT}")
	CHECK_CXX_SYMBOL_EXISTS("pthread_setaffinity_np" "pthread.h" _YACMA_HAS_PTHREAD_SETAFFINITY)
	CHECK_CXX_SYMBOL_EXISTS("pthread_getaffinity_np" "pthread.h" _YACMA_HAS_PTHREAD_GETAFFINITY)
	unset(CMAKE_REQUIRED_LIBRARIES)
	if(_YACMA_HAS_PTHREAD_SETAFFINITY AND _YACMA_HAS_PTHREAD_GETAFFINITY)
		set(YACMA_HAS_PTHREAD_AFFINITY true)
	else()
		set(YACMA_HAS_PTHREAD_AFFINITY false)
	endif()
	if(YACMA_HAS_PTHREAD_AFFINITY)
		message(STATUS "POSIX threads affinity extensions detected.")
	else()
		message(STATUS "POSIX threads affinity extensions not detected.")
	endif()
endif()

if(MINGW AND NOT CMAKE_USE_PTHREADS_INIT)
	# NOTE: the idea here is that the -mthreads flag is useful only when using the native Windows
	# threads, apparently if we are using some pthread variant it is not needed:
	# http://mingw-users.1079350.n2.nabble.com/pthread-vs-mthreads-td7114500.html
	message(STATUS "Native Windows threads detected on MinGW, enabling the -mthreads flag.")
	# NOTE: the -mthreads flag is needed both in compiling and linking. CMake should pass it down
	# to the linker even if only set in CXX_FLAGS.
	set(YACMA_THREADING_CXX_FLAGS "-mthreads")
endif()

message(STATUS "Extra compiler flags for threading: ${YACMA_THREADING_CXX_FLAGS}")

# Detect thread_local availability.
try_compile(YACMA_HAS_THREAD_LOCAL ${CMAKE_BINARY_DIR}
	"${CMAKE_CURRENT_LIST_DIR}/yacma_thread_local_tests.cpp")
if(YACMA_HAS_THREAD_LOCAL)
	message(STATUS "The 'thread_local' keyword is available.")
else()
	message(STATUS "The 'thread_local' keyword is not available.")
endif()

# Mark as included.
set(YACMAThreadingSetupIncluded YES)
