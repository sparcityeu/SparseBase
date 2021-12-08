# Documentation Guide

# Using Doxygen

- Doxyfile is already setup in the root directory of the project
- You can use it as shown below (you can replace firefox with your choice of browser or xdg-open if you are using Linux)

```bash
doxygen
firefox docs/html/index.html
```

# Documenting Code

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

## Documenting Classes/Structs

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

## Documenting Enums

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

## Documenting Functions and Methods

- The line comment is a brief description of the function. For example, these will be used when viewing all the methods of a class. This should not exceed a single line (~80 chars max).
- First part of the block comment is a more detailed description of the function and can be as long as you wish.
- "\param" is used to describe parameters a function takes. It should be followed by the name of the parameter and a short description of it.
- "\return" is used to describe the return value of the function.

```cpp
//! Brief Description.
/*!
	Detailed Description.
  \param a an integer argument.
  \param s a constant character pointer.
  \return The test results
*/
int testMe(int a, const char *s){}
```