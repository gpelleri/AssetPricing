from assetpricing.products.derivative import *


class Option(Derivative):

    def __init__(self, name,
                 expiry,
                 underlying,
                 strike):
        super().__init__(name, expiry, underlying, DerivativeType.OPTION)
        self.strike = strike

    def getStrike(self):
        return self.strike


