from assetpricing.products.equity.option import Option
from assetpricing.models.black_scholes import *
from assetpricing.models.montecarlo import *
from assetpricing.utils.global_types import *
import numpy as np
from assetpricing.products.equity.stock import Stock
import matplotlib.pyplot as plt


class VanillaOption(Option):
    """
    Class that defines the most simple European Option.
    """
    def __init__(self, name, expiry, underlying, strike, option_type: OptionTypes):
        super().__init__(name, expiry, underlying, strike)
        self.option_type = option_type

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


if __name__ == '__main__':
    r = 0.05  # risk-free risk in annual %
    q = 0  # annual dividend rate

    spy = Stock("SPY", False)
    chains = spy.getOptionData()
    surface = spy.build_Impl_Vol_Surface(r, chains, OptionTypes.EUROPEAN_PUT.value)
    # skew = spy.get_Smile(201, r, chains)

    calls = chains[chains["optionType"] == "put"]

    # print the expirations
    set(calls.expiration)
    unique = calls['Expiry']
    # # select an expiration to plot
    calls_at_expiry = calls[calls["expiration"] == "2023-10-20 23:59:59"]
    # # filter out low vols
    filtered_calls_at_expiry = calls_at_expiry[calls_at_expiry.impliedVolatility >= 0.0001]
    # # set the strike as the index so pandas plots nicely
    # display yahoo finance skew
    filtered_calls_at_expiry[["strike", "impliedVolatility"]].set_index("strike").plot(
        title="Implied Volatility Skew", figsize=(7, 4)
    )
    # display computed skew
    skew[["strike", "imp"]].set_index("strike").plot(
        title="Implied Volatility Skew", figsize=(7, 4)
    )
    # we can easily see there's an issue with the high strike, for a reason that i'm missing
    plt.show()


