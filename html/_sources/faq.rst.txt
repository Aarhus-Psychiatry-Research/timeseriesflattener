Frequently Asked Questions
================================


Citing this package
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you wish to use this library in your research, please cite it using (Changing the version if relevant):

.. TODO: the following need to be corrected:
.. code-block::

   @software{Bernstorff_timeseriesflattener_2022,
      author = {Bernstorff, Martin and Enevoldsen, Kenneth and Damgaard, Jakob Grøhn and Hæstrup, Frida and Hansen, Lasse},
      doi = {10.5281/zenodo.7389672},
      month = {11},
      title = {{timeseriesflattener}},
      url = {https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener},
      year = {2022}
   }


Or if you prefer APA:

.. code-block:: 

   Bernstorff, M., Enevoldsen, K., Damgaard, J. G., Hæstrup, F., & Hansen, L. (2022). timeseriesflattener [Computer software]. https://doi.org/10.5281/zenodo.7389672



How do I test the code and run the test suite?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This package comes with an extensive test suite. In order to run the tests,
you'll usually want to clone the repository and build the package from the
source. This will also install the required development dependencies
and test utilities defined in the `pyproject.toml <https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/blob/main/pyproject.toml>`__.


.. code-block:: bash

   poetry install

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
