#!/usr/bin/env bash

# list of test cases you want to run
tests=(
       # test_other_things_on_root_level.py
       test_data.py
       # subpkg_2/test_module_3.py
)

# we must add the module source path because we use `import cs107_package` in our test suite and we
# want to test from the source directly (not a package that we have (possibly) installed earlier)
# pwd -P: Print the current directory, and resolve all symlinks (i.e. show the "physical" path)
export PYTHONPATH="$(pwd -P)/../src":${PYTHONPATH}

python -m pytest ${tests[@]} # will expand all the elements in the tests list
# could also do "$ python -m pytest ."

