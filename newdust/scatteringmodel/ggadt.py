
import scatteringmodel

class GGADT(scatteringmodel.ScatteringModel):
    """
    Child of scatteringModel class

    Attributes
    ----------
    qsca : numpy.ndarray : Scattering efficiency Q (cross-section / geometric cross-section)

    qext : numpy.ndarray : Extinction efficiency Q (cross-section / geometric cross-section)    

    qabs : numpy.ndarray : Absorption efficiency Q (cross-section / geometric cross-section)

    diff : numpy.ndarray : Differential scattering efficiency per steridian

    pars : dict : Parameters from most recent calculation are stored here

    stype : string : A label for the model

    citation : string : A description of how to cite this model
    """
    def __init__(self, from_file):
      """
      Inputs
      ------
      from_file: string: REQUIRED, the name of the fits file with GGADT data in it
      """
      self.read_from_table(from_file)
      self.stype = 'GGADT'
      self.citation = 'https://ui.adsabs.harvard.edu/abs/2016ApJ...817..139H/abstract'

