# // geophysical_notes //

My collection of geophysical notes written as IPython notebooks.


# seismic petrophysics

Do magic things with well log data.

* [Seismic Petrophysics](http://nbviewer.ipython.org/github/aadm/geophysical_notes/blob/master/seismic_petrophysics.ipynb)
* [Seismic Petrophysics / interactive](http://nbviewer.ipython.org/github/aadm/geophysical_notes/blob/master/seismic_petrophysics_interactive.ipynb)

#### support data for "seismic petrophysics"

* `well_2*.txt`: raw log data from Well 2 of [Quantitative Seismic Interpretation (QSI)](https://pangea.stanford.edu/researchgroups/srb/resources/books/quantitative-seismic-interpretation)
* `qsiwell2.csv`: assembled all the logs from various files
* `qsiwell2_frm.csv`: qsiwell2 + fluid replaced elastic logs
* `qsiwell2_augmented.csv`: barebones well data, only Ip, Vp/Vs and LFC (litho-fluid class log)
* `qsiwell2_synthetic.csv`: synthetic data generated through Monte Carlo simulation, same logs as in `qsiwell2_augmented.csv` (Ip, Vp/Vs and LFC)
* `qsiwell2_dataprep.py`: Python script to assemble all the original QSI files


# seismic stuff

How to load and display SEG-Y files, plus some simple ways to play with the data, e.g. extracting amplitude informations, adding noise & filtering. Also, a notebook entirely dedicated to wedge modeling.

* [Seismic data in Python](http://nbviewer.ipython.org/github/aadm/geophysical_notes/blob/master/seismic_data_in_python.ipynb)
* [Amplitude extraction](http://nbviewer.ipython.org/github/aadm/geophysical_notes/blob/master/seismic_amplitude_extraction.ipynb)
* [Wedge modeling for variable angles of incidence](http://nbviewer.ipython.org/github/aadm/geophysical_notes/blob/master/wedge_modeling.ipynb)
* [Notes on spectral decomposition](http://nbviewer.ipython.org/github/aadm/geophysical_notes/blob/master/notes_spec_dec.ipynb)
* [Top Heimdal map, or how to reproduce figure 1 from Avseth et al., 2001](http://nbviewer.ipython.org/github/aadm/geophysical_notes/blob/master/top_heimdal_map.ipynb)
* [AVO projections](http://nbviewer.ipython.org/github/aadm/geophysical_notes/blob/master/avo_projections.ipynb)

#### support data for "seismic stuff"

* `16_81_PT1_PR.SGY`, `16_81_PT2_PR.SGY`, `16_81_PT3_PR.SGY`, `31_81_PR.SGY`: 2D lines in SEGY format from the [USGS Alaska dataset](http://energy.usgs.gov/GeochemistryGeophysics/SeismicDataProcessingInterpretation/NPRASeismicDataArchive.aspx)
* `3d_farstack.sgy`, `3d_nearstack.sgy`: 3D cubes from the QSI dataset (see above)
* `Top_Heimdal_subset.txt`: interpreted horizon for the QSI near and far angle cubes
* `segypy.py`: modified from the [original](https://github.com/rob-smallshire/segpy) to make it self-contained and portable (basically I have inserted `header_definition.py`, `ibm_float.py`, `revisions.py`, `trace_header_definition.py` into `segypy.py`)


## notes on running python

Three options:

1. *CANOPY*

The easiest way is to get Canopy Express, a free download from [Enthought](https://www.enthought.com/products/canopy/), available on all platforms and actively maintained. All the major packages are included as well as many that you will never use; a package manager that takes care of updating all the libraries is also included.

2. *ANACONDA*

A similar solution that I am now using on my work PC; I can't say if this is better than Canopy, there must be differences but to me they are to all practical means equivalent. Get it [here](https://store.continuum.io/cshop/anaconda/).

3. *HOMEMADE SOLUTION*

Finally, for those that feel a little bit more adventurous (and use Macs), there is an homemade solution, which has the advantage of being a barebone installation with minimal impact on disk space; full instructions here: <http://penandpants.com/2013/04/04/install-scientific-python-on-mac-os-x/>. It involves installing [Homebrew](http://brew.sh) on your Mac, which is a [package manager](http://en.wikipedia.org/wiki/Package_manager) that is  a must for anybody tinkering with code and unix-like applications.

After having installed everything enter this command in a Terminal window get IPyhton running and start coding:

    ipython qtconsole

To launch a notebook server to write your own notebooks:

    ipython notebook

Finally, get [Atom](https://atom.io/) to write your code, preview your markdown etc., and you have a minimal (and free!) scientific system.


### using SEG-Y data

To read and write SEG-Y data in Python you need some library like  [ObsPy](https://github.com/obspy/obspy/wiki) or [Segpy](https://github.com/rob-smallshire/segpy/).

I have included in this repo [a modified version of Segpy](https://github.com/aadm/geophysical_notes/blob/master/segypy.py) where I have simply collected all the scattered files of the original module (`segypy.py`, `header_definition.py`, `ibm_float.py`) into a single python file (`segypy.py`).

About ObsPy: I have had trouble in installing it under Canopy Express in Windows since it's not included in the distribution and you have to install it separately. This is how I managed to install it -- first open a Canopy command prompt and type in the following commands:

    easy_install lxml
    easy_install sqlalchemy (*)
    easy_install suds>=0.4
    easy_install flake8
    easy_install nose
    easy_install mock
    easy_install obspy

The package `sqlalchemy` complains about a missing compiler which may render things slower, and suggests to install [mingw](http://www.mingw.org/wiki/Getting_Started); I haven't done this but everything works fine.

Under Mac OS X, first install a [fortran compiler](https://gcc.gnu.org/wiki/GFortranBinaries) (also see [this other option](http://coudert.name/software/gfortran-4.8.2-Mavericks.dmg)). Then: `pip install obspy`.

My above mentioned *HOMEMADE SOLUTION* for a working scientific Python system makes this step however much easier, because the required tools (like that fortran compiler) are all taken care of by Brew (the package manager); so you would just type in `pip install obspy` and that's it.
