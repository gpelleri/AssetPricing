from assetpricing.products.equity.option import Option
from assetpricing.models.black_scholes import *
from assetpricing.models.montecarlo import *
from assetpricing.utils.types import *
import numpy as np
from assetpricing.products.equity.stock import Stock


class VanillaOption(Option):
    """
    Class that defines the most simple European Option.
    """
    def __init__(self, name, expiry, underlying, strike, option_type: OptionTypes):
        super().__init__(name, expiry, underlying, strike)
        self.option_type = option_type

    def getOptionType(self):
        return self.option_type

    def value(self,
              risk_free_rate: float,
              model):

        ul = self.getUnderlying()

        if np.any(ul.getPrice() <= 0.0):
            raise Exception("Stock price must be greater than zero.")

        if np.any(self.getExpiry() < 0.0):
            raise Exception("Time to expiry must be positive.")

        if isinstance(model, BlackScholes):

            value = model.value(ul.getPrice(), self.getStrike(), risk_free_rate, self.getExpiry(), ul.getDiv(),
                                ul.getVol(), self.getOptionType())

        elif isinstance(model, Montecarlo):
            value = model.bs_value_mc(ul.getPrice(), self.getStrike(), risk_free_rate,self.getExpiry(), ul.getDiv(),
                                      ul.getVol(), self.getOptionType())

        else:
            raise Exception("Model : " + model + " isn't implemented")

        return value

