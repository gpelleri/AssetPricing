from assetpricing.products.derivative import *


class Option(Derivative):

    def __init__(self, name,
                 expiry,
                 underlying,
                 strike):
        super().__init__(name, expiry, underlying, DerivativeType.OPTION)
        self.strike = strike
        self.option_type = None

    def get_strike(self):
        return self.strike

    def get_option_type(self):
        return self.option_type


