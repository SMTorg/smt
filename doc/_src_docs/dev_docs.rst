Developer documentation
=======================

This part of the documentation is meant for developers who want to contribute new surrogate models, problem, or sampling methods.

Contributing
------------

Contributing one of these consists of the following steps:

- Fork SMT to make a version of the SMT repo separate from the main one.
- Download *sphinx_auto_embed* by running ``pip install git+https://github.com/hwangjt/sphinx_auto_embed.git``.
- Write the class following the developer API given in the section below, and add it to the right folder, e.g., in smt/surrogate/new_method.py.
- Add the import statement in the corresponding __init__.py file, e.g., smt/surrogate/__init__.py.
- Add tests to the top-level ``tests`` directory following the existing examples.
- Add a documentation page in the appropriate directory, e.g., doc/_src_docs/surrogate/new_method.rstx, using the existing docs as a reference.
- Add an entry in the table of contents so that readers can find the documentation page, e.g., in doc/_src_docs/surrogate.rstx.
- Commit the changes, push to the forked version of SMT and issue a pull request for review and comments from the other developers of SMT and the larger community

Building the documentation
--------------------------

Users can read the docs online at ``smt.readthedocs.io``, but developers who contribute to the docs should build the docs locally to view the output.
This is especially necessary because most of the docs in SMT contain code, code print output, and plots that are dynamically generated and embedded during the doc building process.
The docs are written using reStructuredText, and there are a few custom directives we have added for this embedding of dynamically-generated content.

To build the docs, the user should run go to the ``doc`` directory and run ``sphinx_auto_embed`` and ``make html`` to build the html docs.
This is a 2-step process because ``sphinx_auto_embed`` converts rstx files to rst files and ``make html`` generates the html docs from the rst files.
The landing page for the built docs can then be found at ``doc/_build/html/index.html``, and this is the same page that readers first see when they load ``smt.readthedocs.io``.

Developer API
-------------

.. toctree::
   :maxdepth: 1
   :titlesonly:

   dev_docs/dev_surrogate
   dev_docs/dev_problem
   dev_docs/dev_sampling
