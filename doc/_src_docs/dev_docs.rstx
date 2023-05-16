.. _contributing:

Contributing to SMT
===================

This part of the documentation is meant for developers who want to contribute new surrogate models, problems, or sampling methods.

Contributing to SMT consists of the following steps:

- Fork SMT to make a version of the SMT repo separate from the main one.
- Clone *your* SMT repo and install in development mode: go in your local smt folder and run ``pip install -e .``
- Write the class following the developer API given in the section below, and add it to the right folder, e.g., in ``smt/surrogate_models/``.
- Add the import statement in the corresponding ``__init__.py`` file, e.g., ``smt/surrogate_models/__init__.py``.
- Add tests to the top-level ``tests`` directory following the existing examples and run tests (see `Testing`_ section below) 
- Add a documentation page in the appropriate directory, e.g., ``doc/_src_docs/surrogate_models/rbf.rstx``, using the existing docs as a reference (see `Building the documentation`_ section below).
- Add an entry in the table of contents so that readers can find the documentation page, e.g., in ``doc/_src_docs/surrogate_model.rstx``.
- Test and commit the changes, push to the forked version of SMT and issue a pull request for review and comments from the other developers of SMT and the larger community


Developer API
-------------

.. toctree::
   :maxdepth: 1
   :titlesonly:

   dev_docs/dev_surrogate_models
   dev_docs/dev_problems
   dev_docs/dev_design_space
   dev_docs/dev_sampling_methods


Testing
-------

Install the test runner: ``pip install pytest`` then run: ``pytest`` 

Building the documentation
--------------------------

Users can read the docs online at ``smt.readthedocs.io``, but developers who contribute to the docs should build the docs locally to view the output.
This is especially necessary because most of the docs in SMT contain code, code print output, and plots that are dynamically generated and embedded during the doc building process.
The docs are written using reStructuredText, and there are a few custom directives we have added for this embedding of dynamically-generated content.

First, install *sphinx_auto_embed* by running ``pip install git+https://github.com/hwangjt/sphinx_auto_embed.git``.

To build the docs, the developer should go to the ``doc`` directory and run ``sphinx_auto_embed`` and ``make html`` to build the html docs.
This is a 2-step process because ``sphinx_auto_embed`` converts rstx files to rst files and ``make html`` generates the html docs from the rst files.
The landing page for the built docs can then be found at ``doc/_build/html/index.html``, and this is the same page that readers first see when they load ``smt.readthedocs.io``.
