"""
Author: Dr. Mohamed Amine Bouhlel <mbouhlel@umich.edu>, R. Lafage

This package is distributed under New BSD license.
"""

from smt.surrogate_models import GEKPLS, KPLS, KPLSK, KRG, LS, MGP, QP
from smt.utils.options_dictionary import OptionsDictionary

try:
    from smt.surrogate_models import IDW, RBF, RMTB, RMTC

    COMPILED_AVAILABLE = True
except ImportError:
    COMPILED_AVAILABLE = False


class SurrogateBasedApplication:
    if COMPILED_AVAILABLE:
        _surrogate_type = {
            "KRG": KRG,
            "LS": LS,
            "QP": QP,
            "KPLS": KPLS,
            "KPLSK": KPLSK,
            "GEKPLS": GEKPLS,
            "RBF": RBF,
            "RMTC": RMTC,
            "RMTB": RMTB,
            "IDW": IDW,
            "MGP": MGP,
        }
    else:
        _surrogate_type = {
            "KRG": KRG,
            "LS": LS,
            "QP": QP,
            "KPLS": KPLS,
            "KPLSK": KPLSK,
            "GEKPLS": GEKPLS,
            "MGP": MGP,
        }

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
        >>> from smt.applications import VFM
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
