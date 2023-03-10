from assetpricing.products.equity.option import Option
import numpy as np


class AmericanOption(Option):
    def getOptionType(self):
        return self.option_type

    # TODO : relier au tree value
    def value(self,
              spot_price: float,
              time_to_expiry: float,
              strike_price: float,
              risk_free_rate: float,
              dividendRate: float,
              option_type):

        if np.any(spot_price <= 0.0):
            raise Exception("Stock price must be greater than zero.")

        if np.any(time_to_expiry < 0.0):
            raise Exception("Time to expiry must be positive.")

        return 0