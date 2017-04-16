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
# pyranha packages for Windows.
if is_release_build and not is_python_build:
    print("Non-python release build detected, exiting.")
    sys.exit()

# Get mingw and set the path.
wget(r'https://github.com/bluescarni/binary_deps/raw/master/x86_64-6.2.0-release-posix-seh-rt_v5-rev1.7z', 'mw64.7z')
run_command(r'7z x -oC:\\ mw64.7z', verbose=False)
ORIGINAL_PATH = os.environ['PATH']
os.environ['PATH'] = r'C:\\mingw64\\bin;' + os.environ['PATH']

# Download common deps.
wget(r'https://github.com/bluescarni/binary_deps/raw/master/boost_mingw_64.7z', 'boost.7z')
wget(r'https://github.com/bluescarni/binary_deps/raw/master/nlopt_mingw_64.7z', 'nlopt.7z')
wget(r'https://github.com/bluescarni/binary_deps/raw/master/eigen3.7z', 'eigen3.7z')
# Extract them.
run_command(r'7z x -aoa -oC:\\ boost.7z', verbose=False)
run_command(r'7z x -aoa -oC:\\ nlopt.7z', verbose=False)
run_command(r'7z x -aoa -oC:\\ eigen3.7z', verbose=False)

# Setup of the dependencies for a Python build.
# FIXME
if is_python_build:
    if BUILD_TYPE == 'Python36':
        python_version = '36'
    elif BUILD_TYPE == 'Python35':
        python_version = '35'
    elif BUILD_TYPE == 'Python27':
        python_version = '27'
    else:
        raise RuntimeError('Unsupported Python build: ' + BUILD_TYPE)
    python_package = r'python' + python_version + r'_mingw_64.7z'
    boost_python_package = r'boost_python_' + python_version + r'_mingw_64.7z'
    # Remove any existing Python installation.
    rm_fr(r'c:\\Python' + python_version)
    # Set paths.
    pinterp = r'c:\\Python' + python_version + r'\\python.exe'
    pip = r'c:\\Python' + python_version + r'\\scripts\\pip'
    twine = r'c:\\Python' + python_version + r'\\scripts\\twine'
    pyranha_install_path = r'C:\\Python' + \
        python_version + r'\\Lib\\site-packages\\pyranha'
    # Get Python.
    wget(r'https://github.com/bluescarni/binary_deps/raw/master/' +
         python_package, 'python.7z')
    run_command(r'7z x -aoa -oC:\\ python.7z', verbose=False)
    # Get Boost Python.
    wget(r'https://github.com/bluescarni/binary_deps/raw/master/' +
         boost_python_package, 'boost_python.7z')
    run_command(r'7z x -aoa -oC:\\ boost_python.7z', verbose=False)
    # Install pip and deps.
    wget(r'https://bootstrap.pypa.io/get-pip.py', 'get-pip.py')
    run_command(pinterp + ' get-pip.py --force-reinstall')
    run_command(pip + ' install numpy')
    run_command(pip + ' install mpmath')
    if is_release_build:
        run_command(pip + ' install twine')

# Set the path so that the precompiled libs can be found.
os.environ['PATH'] = os.environ['PATH'] + r';c:\\local\\lib'

# Proceed to the build.
os.makedirs('build')
os.chdir('build')

common_cmake_opts = r'-DCMAKE_PREFIX_PATH=c:\\local -DPAGMO_WITH_EIGEN3=yes -DPAGMO_WITH_NLOPT=yes'

# Configuration step.
if is_python_build:
    # FIXME
    run_command(r'cmake -G "MinGW Makefiles" ..  -DPIRANHA_BUILD_PYRANHA=yes -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-s ' + common_cmake_opts + r' -DBoost_PYTHON_LIBRARY_RELEASE=c:\\local\\lib\\libboost_python' +
                (python_version[0] if python_version[0] == '3' else r'') + r'-mgw62-mt-1_63.dll -DPYTHON_EXECUTABLE=C:\\Python' + python_version + r'\\python.exe -DPYTHON_LIBRARY=C:\\Python' + python_version + r'\\libs\\python' + python_version + r'.dll' +
                r' -DPYTHON_INCLUDE_DIR=C:\\Python' + python_version + r'\\include')
elif 'Debug' in BUILD_TYPE:
    cmake_opts = r'-DCMAKE_BUILD_TYPE=Debug -DPAGMO_BUILD_TESTS=yes -DPAGMO_BUILD_TUTORIALS=yes ' + \
        common_cmake_opts + r' -DCMAKE_CXX_FLAGS_DEBUG="-g0 -Os"'
    run_command(r'cmake -G "MinGW Makefiles" .. ' + cmake_opts)
else:
    raise RuntimeError('Unsupported build type: ' + BUILD_TYPE)

# Build+install step.
# run_command(r'cmake --build . --target install')
run_command(r'mingw32-make install VERBOSE=1 -j2')

# Testing, packaging.
if is_python_build:
    # FIXME
    # Run the Python tests.
    run_command(
        pinterp + r' -c "import pyranha.test; pyranha.test.run_test_suite()"')
    # Build the wheel.
    import shutil
    os.chdir('wheel')
    shutil.move(pyranha_install_path, r'.')
    wheel_libs = 'mingw_wheel_libs_python{}.txt'.format(python_version[0])
    DLL_LIST = [_[:-1] for _ in open(wheel_libs, 'r').readlines()]
    for _ in DLL_LIST:
        shutil.copy(_, 'pyranha')
    run_command(pinterp + r' setup.py bdist_wheel')
    os.environ['PATH'] = ORIGINAL_PATH
    run_command(pip + r' install dist\\' + os.listdir('dist')[0])
    run_command(
        pinterp + r' -c "import pyranha.test; pyranha.test.run_test_suite()"', directory=r'c:\\')
    if is_release_build:
        run_command(twine + r' upload -u bluescarni dist\\' +
                    os.listdir('dist')[0])
elif 'Debug' in BUILD_TYPE:
    run_command(r'ctest')
else:
    raise RuntimeError('Unsupported build type: ' + BUILD_TYPE)
