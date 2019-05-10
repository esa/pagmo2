if(YACMAThreadingSetupIncluded)
    return()
endif()

include(YACMACompilerLinkerSettings)

# Initial thread setup. See FindThreads.cmake for an explanation.
set(THREADS_PREFER_PTHREAD_FLAG YES)
find_package(Threads REQUIRED)
unset(THREADS_PREFER_PTHREAD_FLAG)
message(STATUS "Thread library: ${CMAKE_THREAD_LIBS_INIT}")

# Setup variable for threading-specific flags.
set(YACMA_THREADING_CXX_FLAGS)

# POSIX thread setup. Intended both for UNIX and Windows (the latter when using some sort of
# pthread emulation/wrapper like pthreads-win32).
if(CMAKE_USE_PTHREADS_INIT)
	message(STATUS "POSIX threads detected.")
	# Detect affinity setting primitives.
	include(CheckCXXSymbolExists)
	set(CMAKE_REQUIRED_LIBRARIES "${CMAKE_THREAD_LIBS_INIT}")
	CHECK_CXX_SYMBOL_EXISTS("pthread_setaffinity_np" "pthread.h" _YACMA_HAVE_PTHREAD_SETAFFINITY)
	CHECK_CXX_SYMBOL_EXISTS("pthread_getaffinity_np" "pthread.h" _YACMA_HAVE_PTHREAD_GETAFFINITY)
	unset(CMAKE_REQUIRED_LIBRARIES)
	if(_YACMA_HAVE_PTHREAD_SETAFFINITY AND _YACMA_HAVE_PTHREAD_GETAFFINITY)
		set(YACMA_HAVE_PTHREAD_AFFINITY YES)
	else()
		set(YACMA_HAVE_PTHREAD_AFFINITY NO)
	endif()
	if(YACMA_HAVE_PTHREAD_AFFINITY)
		message(STATUS "POSIX threads affinity extensions detected.")
	else()
		message(STATUS "POSIX threads affinity extensions NOT detected.")
	endif()
endif()

if(MINGW)
	message(STATUS "Enabling the '-mthreads' flag for MinGW.")
	list(APPEND YACMA_THREADING_CXX_FLAGS "-mthreads")
endif()

# Check if we have thread_local.
# NOTE: we need to double check what happens with OSX's clang here.
list(FIND CMAKE_CXX_COMPILE_FEATURES "cxx_thread_local" YACMA_HAVE_THREAD_LOCAL)
if(${YACMA_HAVE_THREAD_LOCAL} EQUAL -1)
	message(STATUS "The 'thread_local' keyword is NOT supported.")
	set(YACMA_HAVE_THREAD_LOCAL NO)
else()
	message(STATUS "The 'thread_local' keyword is supported.")
	set(YACMA_HAVE_THREAD_LOCAL YES)
endif()

# Final report.
if(YACMA_THREADING_CXX_FLAGS)
  message(STATUS "Extra compiler flags for threading: ${YACMA_THREADING_CXX_FLAGS}")
endif()

# Mark as included.
set(YACMAThreadingSetupIncluded YES)
