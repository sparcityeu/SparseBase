# Documentation Guide

## Using Sphinx and Exhale

### Installation

- Documentations are build using [Sphinx](https://www.sphinx-doc.org/en/master/) 
with the [Read The Docs](https://readthedocs.org/) theme.
- We are using [Exhale](https://exhale.readthedocs.io/en/latest/) to embed the 
[Doxygen](https://www.doxygen.nl/index.html) generated C++ documentation into Sphinx.
- As well as having Doxygen installed you also need to install the following python packages:
  - sphinx
  - exhale
  - sphinx-rtd-theme
  - myst_parser
- You can also use the provided `requirements.txt` file in the `docs` directory. But the requirements 
here are frozen to ensure that they work correctly on Read The Docs' build servers.

```bash
pip install -r requirements.txt
```

### Usage

- Documentation can be build using the following commands

```bash
cd docs
make html
```

- This will generate the necessary html files in `_build/html` 
and you can view the documentation by opening the `index.html` file located there.

## Documenting Code

- Doxygen will automatically generate docs from specially marked comments in the code
- For this project, we prefer the QT style marked comments shown below

```cpp
/*!
 * ... text ...
 */

//!
//!... text ...
//!
```

- In general we will try to use the QT commenting styles as much as possible.

### Documenting Classes/Structs

- The line comment is a brief description of the class.
- The block comment is a more detailed description of the class.
- Remember that doxygen will automatically add class hierarchy and methods to the documentation so no need to mention them here.

```cpp
//!  A test class. 
/*!
  A more elaborate class description.
*/
class TestClass {};
```

### Documenting Data Members

- Data members inside classes are commented using a brief description and a detailed description.
- Same pattern as classes

```cpp
class TestClass {
  protected:
    //!  A brief description.
    /*!
      More detailed description.
    */
    int p;
};
```

### Documenting Enums

- Each value and the whole enum must be given a short description as show below.

```cpp
//! Enum Description
enum TEnum { 
 TVal1, /*! Brief Description of TVal1 */
 TVal2, /*! Brief Description TVal2 */
 TVal3  /*! Brief Description TVal3 */
};
```

- In very critical cases the enum itself can be given a more detailed description using the same syntax as a class (see previous section).

### Documenting Functions and Methods

- The line comment is a brief description of the function. For example, these will be used when viewing all the methods of a class. This should not exceed a single line (~80 chars max).
- First part of the block comment is a more detailed description of the function and can be as long as you wish.
- "\param" is used to describe parameters a function takes. It should be followed by the name of the parameter and a short description of it.
- "\tparam" is used to describe templated parameters of a function. It should be followed by the name of the templated parameter and a short description of it.
- "\return" is used to describe the return value of the function.

```cpp
//! Brief Description.
/*!
  Detailed Description.
  \param a an integer argument.
  \param s a constant character pointer.
  \tparam T a templated argument
  \return The test results
*/
template<typename T>
int testMe(int a, const char *s){}
```