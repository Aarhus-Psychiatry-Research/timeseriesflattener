Frequently Asked Questions
================================


Citing this package
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you wish to use this library in your research, please cite the `JOSS paper <https://joss.theoj.org/papers/10.21105/joss.05197.pdf>`__:

.. code-block::

   @article{bernstorff2023timeseriesflattener,
  title={timeseriesflattener: A Python package for summarizing features from (medical) time series},
  author={Bernstorff, Martin and Enevoldsen, Kenneth and Damgaard, Jakob and Danielsen, Andreas and Hansen, Lasse},
  journal={Journal of Open Source Software},
  volume={8},
  number={83},
  pages={5197},
  year={2023}
}


Or if you prefer APA:

.. code-block:: 

   Bernstorff, M., Enevoldsen, K., Damgaard, J., Danielsen, A., & Hansen, L. (2023). timeseriesflattener: A Python package for summarizing features from (medical) time series. Journal of Open Source Software, 8(83), 5197.



How do I test the code and run the test suite?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This package comes with an extensive test suite. In order to run the tests,
you'll usually want to clone the repository and build the package from the
source. This will also install the required development dependencies
and test utilities defined in the `pyproject.toml <https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/blob/main/pyproject.toml>`__.


.. code-block:: bash

   pip install -e ."[dev]"

   python -m pytest


which will run all the test in the `tests` folder.

Specific tests can be run using:

.. code-block:: bash

   python -m pytest tests/desired_test.py



How is the documentation generated?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

timeseriesflattener uses `sphinx <https://www.sphinx-doc.org/en/master/index.html>`__ to generate
documentation. It uses the `Furo <https://github.com/pradyunsg/furo>`__ theme
with custom styling.

To make the documentation you can run:

.. code-block:: bash

   # install sphinx, themes and extensions
   pip install ."[docs,text]"

   # generate html from documentations
   make -C docs html
