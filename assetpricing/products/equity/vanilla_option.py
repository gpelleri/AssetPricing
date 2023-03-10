from assetpricing.products.equity.option import Option
from assetpricing.models.black_scholes import *
from assetpricing.utils.types import *
import numpy as np
from assetpricing.products.equity.stock import Stock


class VanillaOption(Option):
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

        else:
            raise Exception("Model : " + model + " isn't implemented")

        return value


if __name__ == '__main__':
    r = 0.05  # risk-free risk in annual %
    q = 0.02  # annual dividend rate
    # sigma = 0.25  # annual volatility in %
    # steps = 100  # time steps

    # TODO : too much verbose ? Create a constructor with Option as parameter ?
    # TODO Either we make models class based or parameter based
    # TODO : just allow direct access to underlying method ?
    edf = Stock("EDF", True, 100, 0.25, 0.02)
    # edf_call = Option("EDF-call", 0.5, edf, 110)
    edf_call = VanillaOption("EDF-call", 0.5, edf, 110, OptionTypes.EUROPEAN_CALL)
    temp = bs_value(edf_call.getUnderlying().getPrice(),
                    edf_call.getStrike(),
                    r,
                    edf_call.getExpiry(),
                    edf_call.getUnderlying().getDiv(),
                    edf_call.getUnderlying().getVol(),
                    edf_call.getOptionType().value)
    print(temp)
    # 3.8597599507749933
    md = BlackScholes(BlackScholesTypes.ANALYTICAL)
    print(edf_call.value(r, md))
    # print(bs_value(100, 110, r, 1 / 2, q, 0.25, OptionTypes.EUROPEAN_CALL))