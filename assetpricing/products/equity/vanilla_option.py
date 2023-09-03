import pandas as pd

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

        ul = self.get_underlying()

        if np.any(ul.get_price() <= 0.0):
            raise Exception("Stock price must be greater than zero.")

        if np.any(self.get_expiry() < 0.0):
            raise Exception("Time to expiry must be positive.")

        if isinstance(model, BlackScholes):

            value = model.value(ul.get_price(), self.get_strike(), risk_free_rate, self.get_expiry(), ul.get_div(),
                                ul.get_vol(), self.get_option_type())

        elif isinstance(model, Montecarlo):
            value = model.bs_value_mc(ul.get_price(), self.get_strike(), risk_free_rate, self.get_expiry(), ul.get_div(),
                                ul.get_vol(), self.get_option_type())

        else:
            raise Exception("Model : " + model + " isn't implemented")

        return value


if __name__ == '__main__':
    r = 0.05  # risk-free risk in annual %
    q = 0  # annual dividend rate

    apple = Stock("AAPL", False)
    chains = apple.get_option_data()
    chains['Dividend'] = apple.get_div()
    chains['Spot'] = apple.get_price()
    chains['Risk-Free Rate'] = r

    cleaned = apple.clean_option_data(chains)

    x, y, z_imp = apple.build_impl_vol_surface(cleaned, OptionTypes.EUROPEAN_PUT)
    apple.plot_vol_surface(x, y, z_imp)

    # example usage
    spot = apple.get_price()
    expiry = 1
    #imp = apple.get_implied_vol_from_surface(0.66 * spot, expiry, x, y, z_imp)
    #print(imp)
    # local_vol = apple.local_volatility(0.66 * spot, expiry, x, y, z)

    #z_loc = apple.build_local_vol_surface(r, x, y, z_imp)
    #apple.plot_vol_surface(x, y, z_loc)

    z_loc = apple.build_local_vol_surface_v2(r, x, y, z_imp)
    apple.plot_vol_surface(x, y, z_loc)

    # Determine the indices of the nearest discrete strikes
    index_low = np.searchsorted(x[0], 0.66 * spot, side='right') - 1
    index_high = index_low + 1

    # Extract the corresponding strikes and local volatilities
    strike_low = x[0][index_low]
    strike_high = x[0][index_high]
    lv_low = z_loc[index_low]
    lv_high = z_loc[index_high]

    # Perform linear interpolation to get the local volatility at the desired strike
    interp_lv = lv_low + (0.66 * spot - strike_low) * (lv_high - lv_low) / (strike_high - strike_low)

    # The interpolated local volatility at the desired strike
    print(f"Local volatility at spot = {spot} and expiry = {expiry} is {interp_lv:.4f}")
