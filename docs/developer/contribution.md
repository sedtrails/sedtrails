# Contributions

We welcome any kind of contribution to **SedTRAILS** is welcome, from a simple comment or a question, to a full fledged [pull request](https://help.github.com/articles/about-pull-requests/). 

A contribution can be made in the following cases:

- You have a question.
- You think you may have found a bug, including unexpected behavior.
- You want to make changes to the code base to fix a bug, make improvements, or add a new functionality.
- You want to update or documentation to SedTRAILS.

The figure below summarizes the workflow our team follows for developing SedTRAILS. We encourage contributors to adopt it whenever possible. The sections below outline the steps to make a contribution for each of the aforementioned cases.

![sedtrails development workflow](../_static/img/sedtrails-workflow.png)

## A.  You have a question

1. Use the search functionality [here](https://github.com/sedtrails/sedtrails/issues) to see if someone already filed the same issue.
1. If your issue search did not yield any relevant results, open a new issue.
1. Apply the "Question" label. Additionally, apply other labels when relevant.

## B. You think you may have found a bug

1. Use the search functionality [here](https://github.com/sedtrails/sedtrails/issues) to see if someone already filed the same issue.
1. If your issue search did not yield any relevant results, open a new issue and provide enough information to understand the cause and the context of the problem. Depending on the issue, you may also want to include:
    - the [SHA hashcode](https://help.github.com/articles/autolinked-references-and-urls/#commit-shas) of the commit that is causing your problem
    - identifying information (name and version number) for dependencies you're using
    - information about the operating system
    - the steps to reproduce the problem

## C. You want to make changes to the code base

SedTRAILS requires **Python 3.10 or newer**, up to but not including Python 3.14. The minimum is set by Python 3.10 language features used in the codebase.

### Announce your plan

1. (**important**) Announce your plan to other developers *before you start working*. This announcement should be in the form of a (new) issue on the Github repository.
2. (**important**) Wait until a consensus is reached about your idea being a good idea.


### Set up a local development environment to work on your changes

If you are a part of the SedTRAILS team and have write access to the SedTRAILS GitHub repository, skip to the "Develop your contribution" section below. If you are a first-time contributor, follow the below steps:

1. Go to the [SedTRAILS GitHub repository](https://github.com/sedtrails/sedtrails) and click on 'Fork'. This will create a copy of the SedTRAILS repository in your GitHub account. 
            
1. Clone the project to your local computer:
        
    ```bash
    git clone git@github.com:<your-username>/sedtrails.git
    ```

1. Change the directory

    ```bash
    cd sedtrails
    ```


2. Add the upstream repository

    ```bash
    git remote add upstream https://github.com/sedtrails/sedtrails.git
    ```  

3. Now, `git remote -v` will show two remote repositories named:

    * `upstream`, which refers to the SedTRAILS repository 
    * `origin`, which refers to your personal fork

### Develop your contribution

1. Create a branch from the `dev` branch to work on your feature.

    ```bash
    git switch dev
    git checkout -b <my-feature>
    ```  

2. If you are contributing via a fork, make sure to pull in changes from the 'upstream' repository regularly to stay up to date with the `dev` branch on the base repository, while working on your feature branch. Follow the instructions [here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/configuring-a-remote-repository-for-a-fork) and [here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork).

3. Set up a development environment on your computer by installing SedTRAILS in development mode with the following command: (Consider using a virtual environment for this purpose.)

    ```bash
    # On the root of the repository:
    pip install -e .[dev]
    ```
    
4. Set up your code editor to follow [PEP 8](https://peps.python.org/pep-0008/) (remove trailing white space, no tabs, etc.). Maintian a good coding style by checking the code with [Ruff](https://docs.astral.sh/ruff/).

5. Make sure the existing tests pass by running `pytest` from the root of the repository. 

6. Write tests for any new lines of code you add. See [Adding Unit Tests](./adding-unit-tests.md) for advice on this.

7. Include in-code documentation in form of comments and docstrings. Use the [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard) documentation style.

8. Update the user/developer documentation if relevant. Undocumented contributions will not be merged.

### Submitting your contribution

1. Push your feature branch to (your fork of) the SedTRAILS GitHub repository.

2. Create a pull request; for an example, follow the instructions [here](https://help.github.com/articles/creating-a-pull-request/).

### Using keywords in issues and pull requests

- If your pull request completes an issue, you would want to link that pull request to the specific issue it addresses. By doing so, when your pull request is merged, the issue will be closed automatically. To do this, you need to use one of the following keywords in your pull request's description or in a commit message, followed by `#` and the issue number (e.g., `#42`):
  - `close`
  - `closes`
  - `closed`
  - `fix`
  - `fixes`
  - `fixed`
  - `resolve`
  - `resolves`
  - `resolved`
- Note that your pull request must be to the default branch (`dev`). You can read more about it [here](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/linking-a-pull-request-to-an-issue) on GitHub Docs.

## D. You want to improve the SedTRAILS documentation

We use Sphinx and Markdown to write documentation for the SedTRAILS. The root of the documentation is the `docs/` directory. For specific parts, ReStructuredText is also used.

1. [Announce your plan.](https://github.com/sedtrails/sedtrails/blob/dev/CONTRIBUTING.md#announce-your-plan)
2. Follow the same steps to set up a development environment for [making changes to the code base](https://github.com/sedtrails/sedtrails/blob/dev/CONTRIBUTING.md#set-up-a-local-development-environment-to-work-on-your-changes).
3. Install the dependencies in `docs/requirements.txt` using `pip install -r docs/requirements.txt` (Sphinx will also be installed).
4. Update the documentation using Markdown. If unfamiliar with writing Markdown for MyST consult their [guides and documentation](https://myst-parser.readthedocs.io/en/latest/syntax/optional.html).
5. Make sure your contributions are built without errors. Go to the `docs` directory in the terminal with `cd docs/`. Then, build the documentation using `make html`.
6. [Submit your contribution](https://github.com/sedtrails/sedtrails/blob/dev/CONTRIBUTING.md#submitting-your-contribution) for review.

> In case you think you have made a valuable contribution, but you don't know how to write or run tests for it, or how to generate the documentation; don't let this discourage you from making the pull request. The SedTRAILS Team can help you! Just go ahead and submit the pull request. But keep in mind that you might be asked to append additional commits to your pull request.
