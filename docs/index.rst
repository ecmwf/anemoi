.. _index:

####################################
 Welcome to Anemoi's documentation!
####################################

The *Anemoi* framework provides a complete toolkit to develop
data-driven weather models – from data preparation through to inference.
[The development is primarily driven by European Meterological Organisations](https://github.com/ecmwf/anemoi-core/blob/main/CONTRIBUTORS.md#contributing-organisations)
but open to contributions from any organisation or any individual.
The framework is composed of several packages which target the different
components necessary to construct data-driven weather models. To aid
development and deployment, each package collects metadata that can be
used by the subsequent packages. The framework builds upon on
established Python tools including `PyTorch <https://pytorch.org>`_,
`Lighting <https://lightning.ai/pytorch-lightning>`_, `Hydra
<https://hydra.cc>`_, `Zarr <https://zarr.dev>`_, `Xarray
<https://xarray.dev>`_ and `earthkit
<https://earthkit.readthedocs.io/>`_.

.. figure:: _static/anemoi-ecosystem.svg
   :align: center
   :scale: 80 %
   :target: _static/anemoi-ecosystem.svg
   :alt: Figure showing the structure of the Anemoi packages

Possible uses of *Anemoi* are for developing:

-  Global deterministic forecast models
-  Regional deterministic forecast models (limited-area models or
   stretched grids)
-  Coupled atmospheric and ocean models
-  Downscaling models
-  Ensemble/probabilistic models

#########
 What is it for, and what is it not?
#########
Anemoi started out as a codebase for building data-driven weather forecasting models. Since then, it has evolved to include additional *tasks*, such as downscaling (superresolution), time interpolation (temporal downscaling), and autoencoding. What these applications share is the need to build models in which information must be gathered, exchanged, and processed across the spatial dimension. For these use cases, scalable and parallelisable graph neural network tooling is particularly valuable.

Anemoi is not intended to address all Earth-system ML problems. Problems that focus on strictly local networks or on processing long temporal signals (e.g. LSTMs) share little in common with Anemoi’s current focus, and the framework is therefore unlikely to provide significant benefits for those use cases.

#########
 Who is it for?
#########
Anemoi’s primary developers are based in operational meteorological centres, where it is used to create operational data-driven models. Many of these developers are also users of the framework, building and deploying models in practice. This user group places a strong emphasis on structure and reproducibility, in particular the ability to recreate the *same* model as Anemoi evolves and expands.

Anemoi also aims to serve the wider research community by enabling users to test new methods for training data-driven forecasting systems. This user group places greater emphasis on flexibility, allowing ideas to be easily adapted and tested.

#########
 License
#########

*Anemoi* is available under the open source `Apache License`__.

.. __: http://www.apache.org/licenses/LICENSE-2.0.html

####################
 How to Cite Anemoi
####################

If you use Anemoi in your work, we recommend you cite the following
paper as the recommended reference: `Lang, Simon, et al. "AIFS --
ECMWF's Data-Driven Forecasting System." arXiv, 2024`__.

.. __: https://arxiv.org/pdf/2406.01465

BibTeX:

.. code::

   @article{lang2024aifsecmwfsdatadriven,
   title={AIFS -- ECMWF's data-driven forecasting system},
   author={Simon Lang and Mihai Alexe and Matthew Chantry and Jesper Dramsch and Florian Pinault and Baudouin Raoult and Mariana C. A. Clare and Christian Lessig and Michael Maier-Gerber and Linus Magnusson and Zied Ben Bouallègue and Ana Prieto Nemesio and Peter D. Dueben and Andrew Brown and Florian Pappenberger and Florence Rabier},
   year={2024},
   eprint={2406.01465},
   archivePrefix={arXiv},
   primaryClass={physics.ao-ph},
   url={https://arxiv.org/abs/2406.01465}
   }

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   Introduction <self>
   getting-started/package-descriptions.rst
   getting-started/installing.rst
   getting-started/tour.rst

.. toctree::
   :maxdepth: 1
   :caption: Contributing

   contributing/contributing.rst
   contributing/governance.rst
   contributing/roadmap.rst
   contributing/environment.rst
   contributing/guidelines.rst
   contributing/code_style.rst
   contributing/testing.rst
   contributing/documentation.rst

.. toctree::
   :maxdepth: 1
   :caption: Anemoi Package Documentation

   anemoi-datasets <https://anemoi.readthedocs.io/projects/datasets/>
   anemoi-graphs <https://anemoi.readthedocs.io/projects/graphs/>
   anemoi-inference <https://anemoi.readthedocs.io/projects/inference/>
   anemoi-models <https://anemoi.readthedocs.io/projects/models/>
   anemoi-registry <https://anemoi.readthedocs.io/projects/registry/>
   anemoi-training <https://anemoi.readthedocs.io/projects/training/>
   anemoi-transform <https://anemoi.readthedocs.io/projects/transform/>
   anemoi-utils <https://anemoi.readthedocs.io/projects/utils/>
