"""
Author: Dr. Mohamed Amine Bouhlel <mbouhlel@umich.edu>

This package is distributed under New BSD license.
"""

from __future__ import division
from smt.utils.options_dictionary import OptionsDictionary

from smt.methods import LS, QP, KPLS, KRG, KPLSK, GEKPLS
try:
    from smt.methods import IDW, RBF, RMTC, RMTB
    compiled_available = True
except:
    compiled_available = False

class Extensions(object):

    if compiled_available:
        _surrogate_type = {
            'KRG': KRG,'LS': LS,'QP': QP,'KPLS':KPLS,'KPLSK':KPLSK,'GEKPLS':GEKPLS,
            'RBF':RBF,'RMTC':RMTC,'RMTB':RMTB,'IDW':IDW}
    else:
        _surrogate_type = {
            'KRG': KRG,'LS': LS,'QP': QP,'KPLS':KPLS,'KPLSK':KPLSK,'GEKPLS':GEKPLS}

    def __init__(self, **kwargs):
        """
        Constructor where values of options can be passed in.

        For the list of options, see the documentation for the surrogate model being used.

        Parameters
        ----------
        **kwargs : named arguments
            Set of options that can be optionally set; each option must have been declared.

        Examples
        --------
        >>> from smt.extensions import VFM
        >>> extension = VFM(type_bridge = 'Additive', name_model_LF = QP, name_model_bridge =
                           LS, X_LF = xLF, y_LF = yLF, X_HF = xHF, y_HF = yHF, options_LF =
                           dictOptionLFModel, options_bridge = dictOptionBridgeModel)
        """
        self.options = OptionsDictionary()

        self._initialize()
        self.options.update(kwargs)

    def _initialize(self):
        """
        Implemented by the application to declare options and declare what they support (optional).

        Examples
        --------
        self.options.declare('option_name', default_value, types=(bool, int), desc='description')
        """
        pass

    def apply_method(self):
        """
        Run the complete algorithm of the SMT application; e.g.: VFM, ME, EGO...

        """
        self._apply()

    def analyse_results(self, **kwargs):
        """
        Get the final results; e.g., for VFM, two possible analysis are available:
        - kwargs = {x = x, operation = 'predict_values'}
        - kwargs = {x = x, operation = 'predict_derivatives, kx = i}

        """
        return self._analyse_results(**kwargs)
