# Contribution Guidelines

Thank you for choosing to contribute to sparsebase. This document will explain the exact flow of making a contribution. That includes creating an issue, making a pull request, reviewing it, and merging it. 

## Creatin an Issue

Contributions preferably start with an issue on the issue tracker of GitHub. An issue can represent bugs, features, fixes, discussions, or any other addition to the codebase or the documentation.

When creating an issue, you can give it labels. Labels help communicate information about issues like their type, priority, and current state of development. Issues don't necessarily need to have labels. However, labels can make browsing the issue tracker much easier and will provide better visibility for all the contributers to the project.

Labels in this project have three kinds, "type", "state" and "priority". The type of a label is going to be used as its prefix, e.g. "state: pending". The following are the labels used in this project.

1. type: the reason (or contents) of the issue:
    * bug.
    * feature.
    * docs: this includes documenting inside the code (Doxygen and comments), adding documentation pages, and adding examples.
    * discussion: not the same as docs; these are discussions about the project, design decisions, directions of development, etc. It can be the case that the result of a discussion is some added documentation. If that happens, the type should change to docs.
    * chore: mindless work (to an extent.) Changing file names, changing extensions, copying info from README to pages, etc.
    * fix: Iterations on existing features or infrastructure. Optimizations, refactoring, improving API, etc.
    * testing: tests related. 
2. priority: *if* the issue is urgent, how urgent it is:
    * now: now
    * soon: less than now
3. state: the progress of the issue. Note that the “closed” or “done” state is implicitly defined by closing an issue:
    * pending: someone’s working on this issue.
    * review needed: work is done, someone needs to review the work.
    * revision needed: if a reviewer reviews the issue and thinks it’s an issue isn't done yet, they place it here.
    * approved: ready to be merged (and closed)
    * inactive: issue is abandoned for now, but might become active later.
    * wontfix: abandoned indefinetely.

There is also a single label without a kind: "good first issue." This label is for issues that serve as a good starting point for new contributers. 

Note 1: priority labels should only be added to urgent issues; not all issues should have a priority label.

Note 2: it is preferable that an issue have a type and a state, but that is not necessarily the case.

## Contributing to the code 

If you wish to contribute to the code base, you can do so through pull requests.

TL;DR: the process for making a contribution is to make a topic branch out of `origin/develop` into your local machine, make your contributions on this topic branch, push your new branch back into `origin`, and create a pull request to pull your new topic branch into `origin/develop`. DO NOT merge your changes to `develop` on your local machine and push to `origin/develop` directly. 

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
    
5. On GitHub, create a pull request to merge your branch with `develop`; the base of the request will be `develop` and the merging branch will be `feature/<new_feature>`. You can use the same labels used for issues with pull requests. You can also link an issue to your pull request.
6.  Once the contribution is reviewed, a maintainer from the team will merge the pull request into `origin/develop`.

Thank you for your efforts!