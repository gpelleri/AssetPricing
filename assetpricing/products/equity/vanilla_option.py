from assetpricing.products.equity.option import Option
from ...utils.types import *
import numpy as np


class VanillaOption(Option):
    def __init__(self, name, expiry, underlying, strike, option_type):
        super().__init__(name, expiry, underlying, strike)
        self.option_type = option_type

    def getOptionType(self):
        return self.option_type

    def value(self,
              spot_price: float,
              time_to_expiry: float,
              strike_price: float,
              risk_free_rate: float,
              dividendRate: float,
              option_type: OptionTypes):

        if np.any(spot_price <= 0.0):
            raise Exception("Stock price must be greater than zero.")

        if np.any(time_to_expiry < 0.0):
            raise Exception("Time to expiry must be positive.")
