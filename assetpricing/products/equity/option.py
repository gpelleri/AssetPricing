from assetpricing.products.derivative import *


class Option(Derivative):

    def __init__(self, name,
                 expiry,
                 underlying,
                 strike):
        super().__init__(name, expiry, underlying, DerivativeType.OPTION)
        self.strike = strike
        self.option_type = None

    def getStrike(self):
        return self.strike

    def getOptionType(self):
        return self.option_type


