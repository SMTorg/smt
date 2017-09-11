"""
Author: Dr. Mohamed Amine Bouhlel <mbouhlel@umich.edu>

TODO: - check documentation of __init__
      - check documentation of _initialize
      - apply_method
      - analyse_results

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
        >>> from smt.methods import RBF
        >>> sm = RBF(print_global=False)
        """
        self.options = OptionsDictionary()

        self._initialize()
        print self.options['options_LF']
        self.options.update(kwargs)
        print self.options['options_LF']

    def _initialize(self):
        """
        Implemented by the application to declare options and declare what they support (optional).

        Examples
        --------
        self.options.declare('option_name', default_value, types=(bool, int), desc='description')
        self.supports['derivatives'] = True
        """
        pass

    def apply_method(self):
        # TODO some prints options
        self._apply()

    def analyse_results(self, **kwargs):
        # TODO some prints options
        return self._analyse_results(**kwargs)
