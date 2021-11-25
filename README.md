# sparsebase

## Compiling
```
mkdir build && cd build
cmake ..
make
```

This will generate the library as a static library. In addition, the example codes are compiled and their binaries are located in `build/examples`.

## Installation

To install the library, compile the library as shwon in the previous section. Afterwards, you can install the library files either to the systems global location or to a custom location. To install the library to the default system location:
```
cd build
cmake --install .
```

To install the library to a custom directory, do the following:
```
cd build
cmake --install . --prefix "/custom/location"
```

## Usage

When compiling a project that uses SparseBase, simply link the project with the library using the flag `-lsparsebase`.

## Tests

Users can run unit tests easily after building the project. Once its built, do the following:
```
cd build 
ctest -V
```

# Contribution Guidelines

Contributions preferably start with an issue on the issue tracker of GitHub. In addition, a contribution of any kind must be forked out of `origin/develop` and merged back into it. 

TL;DR: the process for making a contribution is to make a topic branch out of `origin/develop` into your local machine, make your contributions on this topic branch, push your new branch back into `origin`, and create a pull request to pull your new topic branch into `origin/develop`. Please do not merge your changes to `develop` on your local machine and push your changes to `origin/develop` directly. 

More precisely, a typical contribution will follow this pattern:

1. Create an issue on GitHub discussing your contribution. At this point, a discussion may happen where the entire team can get on the same page.
2. Pull `origin/develop` into your local to start developing from the latest state of the project, and create a new branch for your contribution. The naming convention for a contribution branch is `feature/<new_feature>`:
    
    ```bash
    # on your local
    cd sparsebase
    git checkout develop
    git pull origin develop
    git checkout -b feature/<new_feature>
    ```
    
3. After you're done working on your feature, make sure that it can be merged cleanly with `origin/develop` by pulling `origin/develop` back into your local machine and merging it with your feature branch:
    
    ```bash
    git checkout develop
    git pull origin develop
    git checkout feature/<new_feature>
    git merge develop
    # merge conflicts may arise
    ```
    
4. Once your feature branch merges successfully with `develop`, push your branch to `origin`:
    
    ```bash
    git checkout feature/<new_feature>
    git push origin feature/<new_feature>
    ```
    
5. On GitHub, create a pull request to merge your branch with `develop`; the base of the request will be `develop` and the merging branch will be `feature/<new_feature>`.
6.  Once the contribution is reviewed, a maintainer from the team will merge the pull request into `origin/develop`.

Thank you for your efforts!
