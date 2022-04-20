# Testing

We use Google Test for carrying out tests. It is a compiled testing library with minimal dependencies. It's integrated into the project using git submodules. We start by going through some Google Test basics. If you're familiar with Google Tests and wish to see how to add tests to our project, skip to the [Adding Tests](#adding-tests) section.

## Terminology

| Term | GTest API | Meaning |
| --- | --- | --- |
| Assertion | ASSERT_*/EXEPECT_* | A single check of the behavior of the program.  |
| Test (case) | TEST | Exercise a particular program using a set of inputs and verify the results. It is made up of one or more assertions. |
| Test suite | TestSuite | A group of related test (cases). |
| Test fixture class | class _ : public ::testing::Test {}; | A class containing objects and routines that can be reused by multiple test cases. |
| Test program |  | A piece of code containing one or more test suites. |

## Assertions

A test is done through one or more *assertions.* An assertion can succeed or fail, and failures are two types depending on the name of the assertion:

1. `EXPECT_*` - nonfatal failure:  a failure that does not terminate the current function
2. `ASSERT_*` - fatal failure: a failure that aborts the current function
    
    Important: this can cause memory leaks since clean-up code doesn't run. Be careful.
    

### Custom message

When an assertion fails, Google Test prints an automated message. Users can add their own message to an assertion using the `operator<<`:

```cpp
ASSERT_EQ(x.size(), y.size()) << "Vectors x and y are of unequal length";
```

### Assertion semantics

Find the full list of assertions [here](https://google.github.io/googletest/reference/assertions.html).

| Assertion | Functionality |
| --- | --- |
| *_EQ(a, b) | Check a and b are equal |

## Test cases

A test case is a C++ function containing a group of assertions. Since it's a function, code within a test case has a state (variables) that can be changed within the test case, but different test cases cannot share states (variables). If all assertions in a test case pass then so does the test case, and if any of them fails then so does the test case.

A test case has a name and belongs to a test suite. We define a test case using the macro `TEST(TestSuiteName, TestName)`.

```cpp
TEST(SparseFormatSuite, CSR){
   // create variables
   // add assertions
   // clean up
}
```

Test cases are grouped by test suite when executing.

## Test Fixtures

When multiple tests might use the same initial state (same starting variables), we can place these variables inside a "Fixture" class that defines how they are constructed and deleted. Then, every test case that needs these variables in this initial state will get a fresh copy of the fixture without any code repetition. Methods can also be added to fixtures for added utility.

Here is an example of a fixture:

```cpp
class QueueTest : public ::testing::Test {
 protected:
  void SetUp() override {
     q1_.Enqueue(1);
     q2_.Enqueue(2);
     q2_.Enqueue(3);
  }

  // void TearDown() override {}

  Queue<int> q0_;
  Queue<int> q1_;
  Queue<int> q2_;
};
```

A test case can use a single fixture only. When a test case function is going to use a fixture `FixtureTest`, you define it using the macro `TEST_F(FixtureTest, TestCaseName)` macro. The test suite name of the test case will be the fixture name. 

When you define a test case using a fixture, you can think of it as creating a sub-class of the fixture, and adding the test case function as a member of the child. Therefore, you have access to all the protected data members of the fixture directly. Here are two test cases using the fixture above.

```cpp
TEST_F(QueueTest, IsEmptyInitially) {
  EXPECT_EQ(q0_.size(), 0);
}

TEST_F(QueueTest, DequeueWorks) {
  int* n = q0_.Dequeue();
  EXPECT_EQ(n, nullptr);

  n = q1_.Dequeue();
  ASSERT_NE(n, nullptr);
  EXPECT_EQ(*n, 1);
  EXPECT_EQ(q1_.size(), 0);
  delete n;

  n = q2_.Dequeue();
  ASSERT_NE(n, nullptr);
  EXPECT_EQ(*n, 2);
  EXPECT_EQ(q2_.size(), 1);
  delete n;
}
```

As you can see, we use the data members `q0_`, `q1_`, and `q2_` directly.

**Important:** each one of these test cases will use *a fresh copy* of the `QueueTest` fixture. They don't affect each other.

### Set up and tear down

Since the fixture is a class, you can define the steps for setting it up inside the constructor, or inside the `SetUp()` function. Similarly, you can define the clean-up code in the destructor or in the `TearDown()` function. For differences between the two, check [here.](https://google.github.io/googletest/faq.html#CtorVsSetUp) 

### Defining a fixture

To create a fixture:

1. Derive a class from `::testing::Test` . Start its body with `protected:`, as
we’ll want to access fixture members from sub-classes.
2. Inside the class, declare any objects you plan to use.
3. If necessary, write a default constructor or `SetUp()` function to prepare
the objects for each test. A common mistake is to spell `SetUp()` as
**`Setup()`** with a small `u` - Use `override` in C++11 to make sure you
spelled it correctly.
4. If necessary, write a destructor or `TearDown()` function to release any
resources you allocated in `SetUp()` . To learn when you should use the
constructor/destructor and when you should use `SetUp()/TearDown()`, read
the [FAQ](https://google.github.io/googletest/faq.html#CtorVsSetUp).
5. If needed, define subroutines for your tests to share.

## Adding tests

Test files follow the same structure as the source files of the library itself. Tests for code in the file `src/path/to/file/<file_name>.cc` are written in the file `tests/suites/path/to/file/<file_name>_tests.cpp`. Each test file will be a separate `CTest` test.

When adding a test for source code in some file, first check to see if that source code file has a matching test file. If it does, add your tests to that test file. Otherwise, create a new test file as shown in the next section and add your tests to that file.

### Adding a new test file

The following are the steps needed to add a new test file. 

#### 1. Determine the directory to which the test file is to be added and check if it exists.

The directory of a test file must match the directory of the source file it is written for, with the `src/` root level directory being replaced by `tests/suites`.  

For example, if the file you are writing tests for is `src/sparsebase/preprocess/preprocess.cc`, then the directory of the test file is `tests/suites/sparsebase/preprocess/`. 

If that subdirectory already exists, you can skip steps 2 and 3. 

#### 2. Create a directory for your test file (if one doesn't already exist).
Assuming that the source file you are writing the tests for is located in `src/path/to/file`, then you need to create a directory that matches that path in the `tests/suites/` folder. 

For example, if the file you are writing tests for is `src/sparsebase/preprocess/preprocess.cc`, then you must create the directory `tests/suites/sparsebase/preprocess/`. 

#### 3. Create a CMakeLists.txt file in that test's directory (if one doesn't already exist).
The `CMakeLists.txt` file will be located in the same directory of the test file (the directory determined in step 1). It will contain the instructions for building the test executable as well as the test definition for CMake. 

After creating the `CMakeLists.txt` file, you must add it to the root level tests `CMakeLists.txt` file. At the end fo the `tests/CMakeLists.txt` file, add the directory you had just created as a subdirectory. 

For example, if the directory created for the test file is `tests/suites/sparsebase/preprocess`, then you would create the empty file `tests/suites/sparsebase/preprocess/CMakeLists.txt` and add the following line to the end of the file `tests/CMakeLists.txt`:
```CMake
add_subdirectory(sparsebase/preprocess)
```

#### 3. Create the tests source file.

The file you create will be located in the test directory determined in step 1. It will have the same name as the source file that it is testing, with the addition of the suffix `_tests`.

For example, if the source file you are testing is `sparsebase/src/sparsebase/preprocess/preprocess.cc`, you will create the file `tests/suites/src/sparsebase/preprocess/preprocess_tests.cc`. 

You must start this file by including the GTest header, the `sparsebase/config.h` header, and then whichever headers you need for the test. For example:

```cpp
#include "gtest/gtest.h"

// needed for CMake flags (e.g. CUDA, PIGO, etc.)
#include "sparsebase/config.h"

#include "sparsebase/format/format.h"
#include "sparsebase/object/object.h"

// test suites and test functions
```

Your tests will follow these headers.
#### 4. Add an executable target and a test command to the 'CMakeLists.txt' file in the test file directory.

Each test file will be compiled into an executable target, and each test executable will be added as a CTest test. The name of the CTest test of a test file is its path relative to the `tests/suites` directory, followed by its name, with the file delimiter being the underscore character. The name of the executable of a test file will be the name of its CTest test with the extension `.test`. For instance, the test file `tests/suites/src/sparsebase/preprocess/preprocess_tests.cc` will have the CTest test name:

```CMake
sparsebase_preprocess_preprocess_tests
```

And will have the executable name:

```CMake
sparsebase_preprocess_preprocess_tests
```

To add the executable and CTest test to the build system generation process, add the following lines to the `CMakeLists.txt` file in the test file's directory:
```CMake
# Add the executable target
add_executable(<test_executable_name> <test_source_file>)
target_link_libraries(<test_executable_name> sparsebase)
target_link_libraries(<test_executable_name> gtest gtest_main)

# Add the CTest test 
add_test(NAME <CTest_test_name> COMMAND <test_executable_name>)
```

For example, given that the test file we are adding is `tests/suites/src/sparsebase/preprocess/preprocess_tests.cc`, we would add the following lines to the file `tests/suites/src/sparsebase/preprocess/CMakeLists.txt`:
```CMake
# Add the executable target
add_executable(sparsebase_preprocess_preprocess_tests.test preprocess_tests.cc)
target_link_libraries(sparsebase_preprocess_preprocess_tests.test sparsebase)
target_link_libraries(sparsebase_preprocess_preprocess_tests.test gtest gtest_main)

# Add the CTest test 
add_test(NAME sparsebase_preprocess_preprocess_tests COMMAND sparsebase_preprocess_preprocess_tests.test)
```

### What about CMake options and flags?
`SparseBase` has many compilation options ranging from including different architectures (e.g. `CUDA`) to different I/O library support (e.g. `PIGO`). We incorporate these flags into testing in the same way we do in the compilation of the library: 
- If the entirety of a test file is conditional on an option, we make its executable and test decleration conditional on that option at the `CMakeLists.txt` level.
- If some tests within a test file are conditional on an option, we use preprocessor directives inside the test source code to control these tests.

The following subsections elaborate on these two approaches.
#### Conditional compilation

If there is a certain test file that is conditional on a CMake option, then you can simply surround the executable and test declerations of that test (shown in step 4 of "Adding a new test file") with an `if()` call contingent on that option.  

For example, the test file `tests/suites/sparsebase/preprocess/cuda/preprocess_tests.cu` should only be added when the library is compiled with the `CUDA` option enabled. Here is how we define its executable and tests in the file `tests/suites/sparsebase/preprocess/cuda/CMakeLists.txt`:
```CMake
if(${CUDA})
    add_executable(sparsebase__preprocess_cuda_preprocess_tests.test preprocess_tests.cu)
    target_link_libraries(sparsebase_preprocess_cuda_preprocess_tests.test sparsebase)
    target_link_libraries(sparsebase_preprocess_cuda_preprocess_tests.test gtest gtest_main)

    add_test(NAME sparsebase_preprocess_preprocess_cuda_tests COMMAND sparsebase_preprocess_preprocess_cuda_tests.test)
endif()
```

#### 2. Conditional tests within a test file

If a certain code (or test) within a test file is conditional on a CMake option, you can surround that code with a preprocessor if statement. Note: make sure that the test file includes the `sparsebase/config.h` file since that is where all the CMake flags are defined.

For example, if a certain test is only required when the library is compiled with `CUDA` enabled, we can define that test in the code as follows:

```C++
#include "sparsebase/config.h"

#ifdef CUDA
TEST(CudaTestSuite, CudaInitialization){
  // code
}
#endif
```

# Running tests

Users can run unit tests easily after building the project. To do so, they must configure CMake to compile tests:

```bash
mkdir build 
cd build
cmake -DRUN_TESTS=ON ..
make
```

Once its built, while in the build directory, do the following:

```
ctest -V
```