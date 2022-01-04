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

For `SparseBase`, we will have the test files following the file structure of the library itself. Tests for a module in file `sparsebase/src/<file_name>.cpp`, the tests of that file will be located in `tests/suites/<file_name>_tests.cpp`. Each test file will be a separate `CTest` test.

When adding a test for some module, first check to see if that module has a test file or not. If it does, add your tests there. Otherwise, create a new test file and modify the `CMakeLists.txt` files as shown in the next section.

### Adding a new test file

When adding a test file, there are three steps you need to follow:

#### 1. Create the '.cpp' file.

Given that the module you are unit-testing is in the file `sparsebase/src/path/package.cpp`, create the file `tests/suites/path/package_tests.cpp`. 

 In this file, you must add the GTest headers as well as whichever headers you need for the test. For example:

```cpp
#include "gtest/gtest.h"
#include "sparsebase/sparse_format.hpp"
#include "sparsebase/sparse_object.hpp"

// test suites and test functions
```

Your tests will follow these headers.

#### 2. Add an executable target to the 'CMakeLists.txt' file in the 'tests' top-level directory.

To add a target for the file we defined in step 1, add the following lines to the end of the file `tests/CMakeLists.txt`

```cpp
add_executable(package_tests.test suites/path/package_tests.cpp)
target_link_libraries(package_tests.test sparsebase)
target_link_libraries(package_tests.test gtest gtest_main)
```

The name of the target should match the name of the test file, but its extension will be `.test` instead of `.cpp`. In our case, given the test file `tests/suites/path/package_tests.cpp`, the target for it is called `package_tests.test`. We add the file executable target for CMake to compile, and we tell CMake that it needs to be linked with the `SparseBase` library and with Google Test. 

### 3. Add the executable target as a test in the 'CMakeLists.txt' file in the root directory

To add the executable as a test, add the following to the `CMakeLists.txt` file in the root directory between the the `if( RUN_TESTS ) ... endif()` tags.

```bash
if ( RUN_TESTS )

	add_test(NAME package_tests COMMAND tests/package_tests.test)

endif()
```

the test name matches the file name, but it doesn't have an extension.

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