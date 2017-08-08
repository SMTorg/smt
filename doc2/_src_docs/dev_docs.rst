Developer documentation
=======================

This part of the documentation is meant for developers who want to contribute new surrogate models, problems, or sampling methods.

Contributing
------------

Contributing one of these consists of the following steps:

- Fork SMT to make a version of the SMT repo separate from the main one
- Write the class following the developer API given in the section below, and add it to the right folder, e.g., in smt/methods/new_method.py.
- Add the import statement in the corresponding __init__.py file, e.g., smt/methods/__init__.py.
- Add tests to the top-level ``tests`` directory following the existing examples.
- Add a documentation page in the appropriate directory, e.g., doc2/_src_docs/methods/new_method.rstx, using the existing docs as a reference.
- Add an entry in the table of contents so that readers can find the documentation page, e.g., in doc2/_src_docs/methods.rstx.
- Commit the changes, push to the forked version of SMT and issue a pull request for review and comments from the other developers of SMT and the larger community

Building the documentation
--------------------------

Users can read the docs online at ``smt.readthedocs.io``, but developers who contribute to the docs should build the docs locally to view the output.
This is especially necessary because most of the docs in SMT contain code, code print output, and plots that are dynamically generated and embedded during the doc building process.
The docs are written using reStructuredText, and there are a few custom directives we have added for this embedding of dynamically-generated content.

To build the docs, the user should run go to the ``doc`` directory and run ``make html`` to build the html docs.
The landing page for the built docs can then be found at ``doc/_build/html/index.html``, and this is the same page that readers first see when they load ``smt.readthedocs.io``.

Developer API
-------------

.. toctree::
   :maxdepth: 1
   :titlesonly:

   dev_docs/dev_methods
   dev_docs/dev_problems
   dev_docs/dev_sampling
