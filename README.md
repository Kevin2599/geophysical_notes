# // geophysical_notes //

My collection of geophysical notes written as Jupyter notebooks.


# seismic petrophysics

Do magic things with well log data.

* [Seismic Petrophysics](http://nbviewer.ipython.org/github/aadm/geophysical_notes/blob/master/seismic_petrophysics.ipynb)
* [Seismic Petrophysics / interactive](http://nbviewer.ipython.org/github/aadm/geophysical_notes/blob/master/seismic_petrophysics_interactive.ipynb)
* [Rock physics modeling and templates](http://nbviewer.ipython.org/github/aadm/geophysical_notes/blob/master/rock_physics_modeling.ipynb)


#### support data for "seismic petrophysics"

* `well_2*.txt`: raw log data from Well 2 of [Quantitative Seismic Interpretation (QSI)](https://srb.stanford.edu/quantitative-seismic-interpretation)
* `qsiwell2.csv`: assembled all the logs from various files
* `qsiwell2_frm.csv`: qsiwell2 + fluid replaced elastic logs
* `qsiwell2_augmented.csv`: barebones well data, only Ip, Vp/Vs and LFC (litho-fluid class log)
* `qsiwell2_synthetic.csv`: synthetic data generated through Monte Carlo simulation, same logs as in `qsiwell2_augmented.csv` (Ip, Vp/Vs and LFC)
* `qsiwell2_dataprep.py`: Python script to assemble all the original QSI files


# seismic stuff

How to load and display SEG-Y files, plus some simple ways to play with the data, e.g. extracting amplitude informations, adding noise & filtering. Also, a notebook entirely dedicated to wedge modeling and how to reproduce a couple of figures from scientific publications.

* [Seismic data in Python](http://nbviewer.ipython.org/github/aadm/geophysical_notes/blob/master/seismic_data_in_python.ipynb)
* [Amplitude extraction](http://nbviewer.ipython.org/github/aadm/geophysical_notes/blob/master/seismic_amplitude_extraction.ipynb)
* [Wedge modeling for variable angles of incidence](http://nbviewer.ipython.org/github/aadm/geophysical_notes/blob/master/wedge_modeling.ipynb)
* [Notes on spectral decomposition](http://nbviewer.ipython.org/github/aadm/geophysical_notes/blob/master/notes_spec_dec.ipynb)
* [Top Heimdal map, or how to reproduce figure 1 from Avseth et al., 2001](http://nbviewer.ipython.org/github/aadm/geophysical_notes/blob/master/top_heimdal_map.ipynb)
* [AVO projections](http://nbviewer.ipython.org/github/aadm/geophysical_notes/blob/master/avo_projections.ipynb)
* [How to calculate AVO attributes](http://nbviewer.ipython.org/github/aadm/geophysical_notes/blob/master/avo_attributes.ipynb)
* [Elastic Impedance](http://nbviewer.ipython.org/github/aadm/geophysical_notes/blob/master/elastic_impedance.ipynb)
* ["The relationship between reflectivity and elastic impedance", or how to reproduce figure 5.62 from Seismic Amplitude by Simm & Bacon (2014)](http://nbviewer.ipython.org/github/aadm/geophysical_notes/blob/master/relationship-reflectivity-elastic-impedance_Simm-Bacon.ipynb)
* [Notes on anisotropic AVO equations](http://nbviewer.ipython.org/github/aadm/geophysical_notes/blob/master/anisotropic_avo.ipynb)
* [AVO Explorer v2: Interactive AVO and AVO classes explorer](http://nbviewer.ipython.org/github/aadm/geophysical_notes/blob/master/avo_explorer_v2.ipynb): meant to be downloaded and run locally.


#### support data for "seismic stuff"

* `16_81_PT1_PR.SGY`, `16_81_PT2_PR.SGY`, `16_81_PT3_PR.SGY`, `31_81_PR.SGY`: 2D lines in SEGY format from the [USGS Alaska dataset](http://energy.usgs.gov/GeochemistryGeophysics/SeismicDataProcessingInterpretation/NPRASeismicDataArchive.aspx)
* `3d_farstack.sgy`, `3d_nearstack.sgy`: 3D cubes from the QSI dataset (see above)
* `Top_Heimdal_subset.txt`: interpreted horizon for the QSI near and far angle cubes

# miscellaneous

Other notebook of interest, maybe only tangentially related to geophysics, such as a notebook showing a comparison between colormaps (the dreadful _jet_ against a bunch of better alternatives) and another that uses the well known Gardner's equation as an excuse to practice data fitting in Python.

* [Color palettes](http://nbviewer.ipython.org/github/aadm/geophysical_notes/blob/master/colormaps.ipynb)
* [Inverse Gardner](http://nbviewer.ipython.org/github/aadm/geophysical_notes/blob/master/inverse_gardner.ipynb)



## notes on running python

I would recommend either [Enthought's Canopy Express]((https://www.enthought.com/products/canopy/)) or [Anaconda](https://www.continuum.io/why-anaconda). I am now using Anaconda both on my work PC and my home computer (an Apple laptop) but I have also been happy with Canopy. There must be some difference between the two but for all practical means they seem to me pretty much the same.

There is also a third solution (the _homemade solution_) which works only on Apple computers and not really recommended unless you are a little bit more adventurous. It has the advantage of being a barebone installation with minimal impact on disk space; full instructions here: <http://penandpants.com/2013/04/04/install-scientific-python-on-mac-os-x/>. It involves installing [Homebrew](http://brew.sh) on your Mac (which is a great [package manager](http://en.wikipedia.org/wiki/Package_manager) essential for anybody tinkering with code and unix-like applications on Macs). Then you do everything through [IPython or Jupyter notebooks](http://jupyter.org/), perhaps in conjunction with a modern (and free!) editor like [Atom](https://atom.io/) to write longer codes and preview your markdown.

However, even if you're tight in (drive) space there is an easier solution than the above _homemade_ recipe, and that involves once again the good folks at Continuum that have created [miniconda](http://conda.pydata.org/miniconda.html) -- highly recommended!

### using SEG-Y data

To read and write SEG-Y data in Python you need additional libraries like  [ObsPy](http://obspy.org) or [Segpy](https://github.com/sixty-north/segpy).

ObsPy is capable of reading large (10Gb) SEG-Ys exported from Petrel; it has also matured to v1.0 so I would recommend it now-- if only for its SEG-Y support (what I didn't like before was simply the concept of using a large library aimed at research seismologists that does too many things that I don't use; but yes you could say that's true also for the way I use Numpy!). I haven't tried the latest version of Segpy (which only runs on Python 3) on similarly large datasets.

## TO-DO

* add examples on using `xarray` for 3D seismic cubes.
