# Representing operations in SparseBase

SparseBase allows users to carry out any operation that involve `Format` objects. 
Reordering, partitioning, graph coarsening, matrix factorization, feature extraction, and tensor decomposition are a few examples of such operations.
Operations in SparseBase are implemented in an object-oriented manner; users instantiate objects that they use to carry out operations. 
Each object can contain multiple functions that handle different `Format` types, but all these functions will carry 
out the same operation semantically speaking. 
Every object also comes with a matching and conversion mechanism. 
This mechanism allows an object to take the user input `Format`, find a function that matches its format type, and use that function. If no such function is found, the mechanism converts the input into a different format for which it has a function.
This mechanism has two main benefits:

1. **Users need a single representation only.** Users do not need to worry about the format type of their data. The library will take care of doing the necessary conversions to carry out their work. Simply load your data and pass it to a an operation's object and it will do the work for you.
2. **Developers don’t need to cover the entire format space.** When a developer wishes to add a new functionality, e.g. a new reordering algorithm, they do not need to account for every input format type. A single implementation on a common format type will be sufficient; any object that is convertible to that format type will be able to use this reordering. Of course, if they wish, they could provide additional implementations for different formats.


Since an operation can encapsulate many aspects of sparse data, it is important to keep it as generic as possible. 
We achieve this generality by making each functionality class abstract in terms of:

1. The number of `Format` objects it takes as input.
2. Its return type.
3. Its auxiliary hyperparameters.

This allows developers to implement virtually any type of operation on any number of `Format` objects. 
In addition, it lessens the burden on the users and allows them to explore the functionality space without having to worry about data representation. 
