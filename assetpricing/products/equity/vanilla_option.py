import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

from assetpricing.models.black_scholes import *
from assetpricing.models.montecarlo import *
from assetpricing.products.equity.option import Option
from assetpricing.products.equity.stock import Stock
from assetpricing.utils.global_types import *


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
            value = model.bs_value_mc(ul.getPrice(), self.getStrike(), risk_free_rate, self.getExpiry(), ul.getDiv(),
                                      ul.getVol(), self.getOptionType())

        else:
            raise Exception("Model : " + model + " isn't implemented")

        return value


if __name__ == '__main__':
    r = 0.05  # risk-free risk in annual %
    q = 0  # annual dividend rate

    spy = Stock("SPY", False)
    chains = spy.getOptionData()
    chains['Dividend'] = spy.getDiv()
    chains['Spot'] = spy.getPrice()
    chains['Risk-Free Rate'] = r
    skew_df = spy.build_Impl_Vol_Surface(r, chains[['lastPrice', 'strike', 'Expiry', 'OptionType',
                                         'impliedVolatility', 'Dividend', 'Spot', 'Risk-Free Rate']],
                                         OptionTypes.EUROPEAN_PUT.value)
    # create a grid of x, y, and z values
    y = skew_df['Expiry'].values
    x = skew_df['strike'].values
    z = skew_df['imp'].values

    # create a grid of x and y values for the plot
    xi = np.linspace(min(x), max(x), 100)
    yi = np.linspace(min(y), max(y), 100)
    xi, yi = np.meshgrid(xi, yi)

    # interpolate the z values onto the grid of x and y values
    zi = griddata((x, y), z, (xi, yi), method='linear')

    # plot the surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # x, y, and z are arrays of Expiry, Strike, and Implied Volatility values, respectively
    ax.plot_surface(xi, yi, zi)

    ax.set_ylabel('Expiry')
    ax.set_xlabel('Strike')
    ax.set_zlabel('Implied Volatility')

    plt.show()
