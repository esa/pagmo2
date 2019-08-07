import os
import re
import sys


def wget(url, out):
    import urllib.request
    print('Downloading "' + url + '" as "' + out + '"')
    urllib.request.urlretrieve(url, out)


def rm_fr(path):
    import os
    import shutil
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


def run_command(raw_command, directory=None, verbose=True):
    # Helper function to run a command and display optionally its output
    # unbuffered.
    import shlex
    import sys
    from subprocess import Popen, PIPE, STDOUT
    print(raw_command)
    proc = Popen(shlex.split(raw_command), cwd=directory,
                 stdout=PIPE, stderr=STDOUT)
    if verbose:
        output = ''
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            line = str(line, 'utf-8')
            # Don't print the newline character.
            print(line[:-1])
            sys.stdout.flush()
            output += line
        proc.communicate()
    else:
        output = str(proc.communicate()[0], 'utf-8')
    if proc.returncode:
        raise RuntimeError(output)
    return output


# Build type setup.
BUILD_TYPE = os.environ['BUILD_TYPE']
is_release_build = (os.environ['APPVEYOR_REPO_TAG'] == 'true') and bool(
    re.match(r'v[0-9]+\.[0-9]+.*', os.environ['APPVEYOR_REPO_TAG_NAME']))
if is_release_build:
    print("Release build detected, tag is '" +
          os.environ['APPVEYOR_REPO_TAG_NAME'] + "'")
is_python_build = 'Python' in BUILD_TYPE

# Just exit if this is a release build but not a Python one. The release of the source code
# is done in travis, from appveyor we manage only the release of the
# pygmo packages for Windows.
if is_release_build and not is_python_build:
    print("Non-python release build detected, exiting.")
    sys.exit()

# Get mingw and set the path.
# USING: mingw64 8.1.0 from the appveyor VMs
run_command(
    r'mv C:\\mingw-w64\\x86_64-8.1.0-posix-seh-rt_v6-rev0\\mingw64 C:\\mingw64')
ORIGINAL_PATH = os.environ['PATH']
os.environ['PATH'] = r'C:\\mingw64\\bin;' + os.environ['PATH']

# Download common deps.
wget(r'https://github.com/bluescarni/binary_deps/raw/master/boost_mgw81-mt-x64-1_70.7z', 'boost.7z')
wget(r'https://github.com/bluescarni/binary_deps/raw/master/nlopt_mingw81_64.7z', 'nlopt.7z')
wget(r'https://github.com/bluescarni/binary_deps/raw/master/eigen3.7z', 'eigen3.7z')
wget(r'https://github.com/bluescarni/binary_deps/raw/master/tbb_2019_mgw81.7z', 'tbb.7z')
# Extract them.
run_command(r'7z x -aoa -oC:\\ boost.7z', verbose=False)
run_command(r'7z x -aoa -oC:\\ nlopt.7z', verbose=False)
run_command(r'7z x -aoa -oC:\\ eigen3.7z', verbose=False)
run_command(r'7z x -aoa -oC:\\ tbb.7z', verbose=False)

# Setup of the dependencies for a Python build.
if is_python_build:
    if '64_Python37' in BUILD_TYPE:
        python_version = r'37'
        python_folder = r'Python37-x64'
        python_library = r'C:\\' + python_folder + r'\\python37.dll '
    elif '64_Python36' in BUILD_TYPE:
        python_version = '36'
        python_folder = r'Python36-x64'
        python_library = r'C:\\' + python_folder + r'\\python36.dll '
    elif '64_Python27' in BUILD_TYPE:
        python_version = r'27'
        python_folder = r'Python27-x64'
        python_library = r'C:\\' + python_folder + r'\\libs\\python27.dll '
        # For py27 I could not get it to work with the appveyor python (I was close but got tired).
        # Since this is anyway going to disappear (py27 really!!!), I am handling it as a one time workaround using the old py27 patched by bluescarni
        rm_fr(r'c:\\Python27-x64')
        wget(r'https://github.com/bluescarni/binary_deps/raw/master/python27_mingw_64.7z', 'python.7z')
        run_command(r'7z x -aoa -oC:\\ python.7z', verbose=False)
        run_command(r'mv C:\\Python27 C:\\Python27-x64', verbose=False)
    else:
        raise RuntimeError('Unsupported Python build: ' + BUILD_TYPE)

    # Set paths.
    pinterp = r"C:\\" + python_folder + r'\\python.exe'
    pip = r"C:\\" + python_folder + r'\\scripts\\pip'
    twine = r"C:\\" + python_folder + r'\\scripts\\twine'
    pygmo_install_path = r"C:\\" + python_folder + r'\\Lib\\site-packages\\pygmo'
    # Install pip and deps.
    wget(r'https://bootstrap.pypa.io/get-pip.py', 'get-pip.py')
    run_command(pinterp + ' get-pip.py --force-reinstall')
    # NOTE: at the moment we have troubles installing ipyparallel.
    # Just skip it.
    # run_command(pip + ' install numpy cloudpickle ipyparallel')
    run_command(pip + ' install numpy cloudpickle dill')
    if is_release_build:
        run_command(pip + ' install twine')

# Set the path so that the precompiled libs can be found.
os.environ['PATH'] = os.environ['PATH'] + r';c:\\local\\lib'

# Proceed to the build.
# NOTE: at the moment boost 1.70 seems to have problem to autodetect
# the mingw library (with CMake 3.13 currently installed in appveyor)
# Thus we manually point to the boost libs.
common_cmake_opts = r'-DCMAKE_PREFIX_PATH=c:\\local ' + \
                    r'-DCMAKE_INSTALL_PREFIX=c:\\local ' + \
                    r'-DBoost_INCLUDE_DIR=c:\\local\\include ' + \
                    r'-DBoost_SERIALIZATION_LIBRARY_RELEASE=c:\\local\\lib\\libboost_serialization-mgw81-mt-x64-1_70.dll '


# Configuration step.
if is_python_build:
    os.makedirs('build_pagmo')
    os.chdir('build_pagmo')
    run_command(r'cmake -G "MinGW Makefiles" .. ' +
                common_cmake_opts +
                r'-DPAGMO_WITH_EIGEN3=yes ' +
                r'-DPAGMO_WITH_NLOPT=yes ' +
                r'-DCMAKE_BUILD_TYPE=Release ')
    run_command(r'mingw32-make install VERBOSE=1 -j2')
    # Alter the path to find the pagmo dll.
    os.environ['PATH'] = os.getcwd() + ";" + os.environ['PATH']
    os.chdir('..')
    os.makedirs('build_pygmo')
    os.chdir('build_pygmo')
    run_command(r'cmake -G "MinGW Makefiles" .. ' +
                common_cmake_opts +
                r'-DPAGMO_BUILD_PYGMO=yes ' +
                r'-DPAGMO_BUILD_PAGMO=no ' +
                r'-DCMAKE_BUILD_TYPE=Release ' +
                r'-DCMAKE_CXX_FLAGS=-s ' +
                r'-DBoost_PYTHON' + python_version + r'_LIBRARY_RELEASE=c:\\local\\lib\\libboost_python' + python_version + r'-mgw81-mt-x64-1_70.dll ' +
                r'-DPYTHON_INCLUDE_DIR=C:\\' + python_folder + r'\\include ' +
                r'-DPYTHON_EXECUTABLE=C:\\' + python_folder + r'\\python.exe ' +
                r'-DPYTHON_LIBRARY=' + python_library +
                r'-DCMAKE_CXX_FLAGS="-D_hypot=hypot"')
    run_command(r'mingw32-make install VERBOSE=1 -j2')
elif 'Debug' in BUILD_TYPE:
    os.makedirs('build_pagmo')
    os.chdir('build_pagmo')
    run_command(r'cmake -G "MinGW Makefiles" .. ' +
                common_cmake_opts +
                r'-DPAGMO_WITH_EIGEN3=yes ' +
                r'-DPAGMO_WITH_NLOPT=yes ' +
                r'-DCMAKE_BUILD_TYPE=Debug ' +
                r'-DPAGMO_BUILD_TESTS=yes ' +
                r'-DPAGMO_BUILD_TUTORIALS=yes ' +
                r'-DBoost_UNIT_TEST_FRAMEWORK_LIBRARY_RELEASE=c:\\local\\lib\\libboost_unit_test_framework-mgw81-mt-x64-1_70.dll ' +
                r'-DCMAKE_CXX_FLAGS_DEBUG="-g0 -Os"')
    run_command(r'mingw32-make install VERBOSE=1 -j2')
    # Alter the path to find the pagmo dll.
    os.environ['PATH'] = os.getcwd() + ";" + os.environ['PATH']
    run_command(r'ctest')
else:
    raise RuntimeError('Unsupported build type: ' + BUILD_TYPE)

# Packaging.
if is_python_build:
    # Run the Python tests.
    run_command(
        pinterp + r' -c "import pygmo; pygmo.test.run_test_suite(1)"')
    # Build the wheel.
    import shutil
    os.chdir('wheel')
    shutil.move(pygmo_install_path, r'.')
    wheel_libs = 'mingw_wheel_libs_python{}.txt'.format(python_version)
    DLL_LIST = [_[:-1] for _ in open(wheel_libs, 'r').readlines()]
    for _ in DLL_LIST:
        shutil.copy(_, 'pygmo')
    run_command(pinterp + r' setup.py bdist_wheel')
    os.environ['PATH'] = ORIGINAL_PATH
    run_command(pip + r' install dist\\' + os.listdir('dist')[0])
    run_command(
        pinterp + r' -c "import pygmo; pygmo.test.run_test_suite(1)"', directory=r'c:\\')
    if is_release_build:
        run_command(twine + r' upload -u ci4esa dist\\' +
                    os.listdir('dist')[0])
