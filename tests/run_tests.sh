#!/usr/bin/env bash

# list of test cases you want to run
tests=(
       # test_other_things_on_root_level.py
       test_data_included.py
       test_data_not_included.py
)

# we must add the module source path because we use `import cs107_package` in our test suite and we
# want to test from the source directly (not a package that we have (possibly) installed earlier)
# pwd -P: Print the current directory, and resolve all symlinks (i.e. show the "physical" path)
export PYTHONPATH="$(pwd -P)/../src":${PYTHONPATH}

# look up how to pass an argument to pytest
# both .py files use 'train_relative_path = "../data/raw/train.csv"'

# if we call "bash run_tests.sh include" from Terminal, then will run both tests
# if we call "bash run_tests.sh" from Terminal, will only run test with the data included
if [ $# -ge 1 ]; then
       python -m pytest ${tests[@]} # will expand all the elements in the tests list
else
       python -m pytest ${tests[0]} # will only run first element of tests list
fi

