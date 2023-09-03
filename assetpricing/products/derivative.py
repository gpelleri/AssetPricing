from assetpricing.products.security import *
from assetpricing.utils.global_types import *
from assetpricing.products.underlying import Underlying

# TODO : remove pre-determined type on underlying
# create another class, underlying, that allows to have access to methods according to it's initial given object ?


class Derivative(Security):
    def __init__(self, name, expiry, underlying: Underlying, derivative_type):
        """
        Create a derivative for a given underlying. Derivatives provide an interface for options, swaps etc
        :param name: derivatives name - usually yahoo finance ticker
        :param expiry: derivative expiry date
        :param underlying: underlying object
        """
        super().__init__(name, SecurityType.DERIVATIVE.value)
        self._expiry = expiry  # in years
        self._underlying = underlying  # Only stock implemented for now but this allows for more flexibility later
        self.deri_type = derivative_type  # equity option , swap, fx der etc ..

    def get_expiry(self):
        return self._expiry

    def get_underlying(self):
        return self._underlying

    def get_derivative_type(self):
        return self.deri_type

