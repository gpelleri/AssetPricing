import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import griddata, Rbf

from assetpricing.products.security import *
from ...utils.global_types import *
from assetpricing.utils.vol_utils import *
import yfinance as yf
import pandas as pd
import datetime as dt


class Stock(Security):
    def __init__(self,
                 name,
                 is_fictive: bool,
                 price=0,
                 vol=0,
                 div=0,
                 ):
        """
        Class to create a stock
        :param name: stock name - usually put yahoo finance ticker
        :param is_fictive : flag to indicate if stock is fictive and if data should be downloaded from yahooFinance
        :param price: stock price - spot price
        :param vol: implied annual volatility - can be computed through getImpliedVol
        """
        super().__init__(name, SecurityType.STOCK.value)
        if is_fictive:
            self.price = price  # given stock price
            self._vol = vol  # given stock volatility
        else:
            self.ticker = name
            self.download_price(name)  # set price to current spot price

        self.div = div
        self._derivatives = None

    def get_name(self):
        return self.name

    def get_vol(self):
        return self._vol

    def get_div(self):
        return self.div

    def get_derivatives(self):
        return self._derivatives

    # TODO probably poor naming
    # What if we want to extract past prices and not only spot ? -> out of scope atm
    def download_price(self, ticker):
        """
        Function to "pull" stock data from yahoo finance and set its price to current spot price
        :param ticker: name of ticker on yahoo finance
        """
        df = yf.download(ticker, period="1d")
        self.price = df['Close'][-1]

    # def createOption(self, option_type, expiry, strike):
    #     """
    #     Create a new option for the stock. If there is current derivatives list, we create one
    #     :param option_type: option type from OptionTypes Class
    #     :param expiry: option expiry date / maturity
    #     :param strike: option strike
    #     """
    #
    #     if option_type == OptionTypes.EUROPEAN_CALL:
    #         option = VanillaOption(self.getName(), expiry, self, strike, option_type)
    #     else:  # if option_type == OptionTypes.EUROPEAN_PUT:
    #         option = VanillaOption(self.getName(), expiry, self, strike, option_type)
    #
    #     if self._derivatives is None:
    #         self._derivatives = []
    #     self._derivatives.append(option)

    def __set_vol__(self, vol):
        """
        Private setter : must ONLY be called by derivatives class when extracting implied vol
        """
        self._vol = vol

    def get_option_data(self):
        asset = yf.Ticker(self.ticker)
        expirations = asset.options

        chains = pd.DataFrame()
        # iterate over expiry dates
        for expiration in expirations:
            # tuple of two dataframes
            opt = asset.option_chain(expiration)

            calls = opt.calls
            calls['optionType'] = "call"
            puts = opt.puts
            puts['optionType'] = "put"

            chain = pd.concat([calls, puts])
            chain['expiration'] = pd.to_datetime(expiration) + pd.DateOffset(hours=23, minutes=59, seconds=59)

            chains = pd.concat([chains, chain])

        chains["daysToExpiration"] = (chains.expiration - dt.datetime.today()).dt.days + 1
        chains["Expiry"] = chains["daysToExpiration"] / 365
        fct = lambda x: OptionTypes.EUROPEAN_CALL.value if x == "call" else OptionTypes.EUROPEAN_PUT.value
        chains["OptionType"] = chains["optionType"].apply(fct)

        return chains

    def clean_option_data(self, df: DataFrame):
        """
        This function is a must called before creating a vol surface. It cleans dataset removing irrelevant data
        based on the following conditions :
        - trade must have occurred within last business days
        - spread must be smaller than expiry spread min + 4 * expiry spread std.
        - volume / open interest has to be valid
        - and obviously we remove out of the money data (Arbitrary conditions for now)
        """

        # # Filter out options with high bid/ask spreads
        df['spread'] = df['ask'] - df['bid']
        spread_mean = df.groupby('Expiry')['spread'].mean()
        spread_std = df.groupby('Expiry')['spread'].std()
        merged = pd.concat([spread_mean, spread_std], axis=1).reset_index()
        merged.columns = ['Expiry', 'mean', 'std']
        df = df.merge(merged, on='Expiry')
        df = df[df['spread'] < (df['mean'] + 4 * df['std'])]

        # Filter out options that haven't been traded in the last 5 business days
        df['lastTradeDate'] = pd.to_datetime(df['lastTradeDate'])
        five_days_ago = pd.Timestamp.now().normalize() - pd.offsets.BDay(5)
        five_days_ago = five_days_ago.tz_localize('UTC')
        #df = df[df['lastTradeDate'] > five_days_ago]

        df = df.drop(columns=['spread', 'mean', 'std'])

        # Filter out options with zero or NaN volume
        df = df[(df['volume'].notna()) & (df['volume'] != 0)]

        # Filter out options with zero or NaN open interest
        df = df[(df['openInterest'].notna()) & (df['openInterest'] != 0)]

        # Filter out options that are in the money
        df = df[df['inTheMoney'] == False]

        # return only interesting cols
        df = df[['lastPrice', 'strike', 'Expiry', 'OptionType',
            'impliedVolatility', 'Dividend', 'Spot', 'Risk-Free Rate']]
        return df

    def get_skew(self, T, df: DataFrame):
        """
        Extract Implied volatilty from options prices for a given maturity T
        """
        # filter the input DataFrame to only contain data for the given `Expiry`
        df_filtered = df[df['Expiry'] == T]

        # filter the DF. We only use put data for now, for which implied vol is sufficient
        puts = df_filtered[(df_filtered['impliedVolatility'] > 0.0001)].copy()

        puts['imp'] = puts.apply(implied_volatility_row, axis=1)
        return puts

    def interpolate_skew(self, df: DataFrame, method='linear'):
        """
        Computes implied volatility interpolation based on implied volatility extracted from options prices
        """

        # filter the DataFrame to only contain 'Expiry' between 0.5 and 2.5
        expiry_filtered = df[(df['Expiry'] >= 0.5) & (df['Expiry'] <= 2.5)]

        # group the filtered DataFrame by 'Expiry' and apply the `get_Skew` function to each group
        expiries = expiry_filtered['Expiry'].unique()
        interpolations = []
        for T in expiries:
            skew_df = self.get_skew(T, expiry_filtered)
            strike = skew_df['strike'].values
            imp = skew_df['imp'].values

            # Use interpolation on skews
            interp_strike = np.linspace(strike[0], strike[-1], 1000)
            if method == 'numpy':
                interp = np.interp(interp_strike, strike, imp)
            elif method == 'cubic':
                interp = griddata(strike, imp, interp_strike, method='cubic')

            # This seems to be the best method
            else:
                interp = griddata(strike, imp, interp_strike, method='linear')

            df_interp = pd.DataFrame({'interpolation': interp})
            df_interp['Expiry'] = T
            df_interp['strike'] = interp_strike
            interpolations.append(df_interp)

        # concatenate
        interpolations = pd.concat(interpolations, ignore_index=True)

        return interpolations

    def interpolate_imp(self, df: DataFrame):
        """
        Uses yahoo finance implied volatilty values as source data
        """

        # filter the DataFrame to only contain 'Expiry' between 0.5 and 2.5
        expiry_filtered = df[(df['Expiry'] >= 0.5) & (df['Expiry'] <= 2.5)]

        # group the filtered DataFrame by 'Expiry' and apply the `get_Skew` function to each group
        expiries = expiry_filtered['Expiry'].unique()
        interpolations = []
        for T in expiries:
            skew_df = self.get_skew(T, expiry_filtered)
            strike = skew_df['strike'].values
            imp = skew_df['impliedVolatility'].values

            # Use interpolation on skews
            interp_strike = np.linspace(strike[0], strike[-1], 1000)
            interp = griddata(strike, imp, interp_strike, method='cubic')

            df_interp = pd.DataFrame({'interpolation': interp})
            df_interp['Expiry'] = T
            df_interp['strike'] = interp_strike
            interpolations.append(df_interp)

        # concatenate
        interpolations = pd.concat(interpolations, ignore_index=True)

        return interpolations

    def build_impl_vol_surface(self, df: DataFrame, option_type: OptionTypes, method='linear'):
        options_filtered = df[df['OptionType'] == option_type.value]

        interp_skews = self.interpolate_skew(options_filtered, method)
        #interp_skews = self.interpolate_imp(options_filtered)

        x = interp_skews['strike'].values
        y = interp_skews['Expiry'].values
        z = interp_skews['interpolation'].values

        # define RBF parameters
        rbf_type = 'cubic'
        epsilon = 2.0
        rbf = Rbf(x, y, z, function=rbf_type, epsilon=epsilon)

        # create a grid of x and y values for the plot
        xi = np.linspace(min(x), max(x), 100)
        yi = np.linspace(min(y), max(y), 100)
        xi, yi = np.meshgrid(xi, yi)

        # evaluate interpolated surface using RBF
        zi = rbf(xi, yi)

        return xi, yi, zi

    def plot_vol_surface(self, x, y, z):
        fig = plt.figure()
        # plot the first surface
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.set_title('Implied Volatility Surface')
        surf1 = ax1.plot_surface(x, y, z, cmap='coolwarm', alpha=0.8)
        ax1.set_xlabel('Strike')
        ax1.set_ylabel('Expiry')
        ax1.set_zlabel('Implied Volatility')
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

        plt.show()

    def get_implied_vol_from_surface(self, S, expiry, xi, yi, zi):
        implied_volatility_1d = zi.ravel()

        implied_volatility_interp = griddata((xi.ravel(), yi.ravel()), implied_volatility_1d,
                                             (S, expiry), method='cubic')

        return implied_volatility_interp

    def local_volatility(self, spot, expiry, xi, yi, zi):
        """
        Not working : local vol is too small : bad interpolation on d_k
        """
        # compute the local volatility using Dupire's formula
        delta_k = 0.1
        vol1 = self.get_implied_vol_from_surface(spot - delta_k, expiry, xi, yi, zi)
        vol2 = self.get_implied_vol_from_surface(spot, expiry, xi, yi, zi)
        vol3 = self.get_implied_vol_from_surface(spot + delta_k, expiry, xi, yi, zi)

        delta_t = 0.01
        vol_dt = self.get_implied_vol_from_surface(spot, expiry - delta_t, xi, yi, zi)
        vol_dt2 = self.get_implied_vol_from_surface(spot, expiry + delta_t, xi, yi, zi)
        # Looks fine
        d_vol_d_t = (vol_dt - vol_dt2) / (delta_t * 2)
        # too small or even negative d_vol_d_k ??
        d_vol_d_k = (vol1 - 2 * vol2 + vol3) / (delta_k ** 2)
        local_vol = np.sqrt(2 * d_vol_d_t / (spot ** 2) * d_vol_d_k)
        return local_vol

    def build_local_vol_surface(self, r, xi, yi, zi):
        # Step 1: Calculate first derivatives
        dzi_dx = np.gradient(zi, xi[0], axis=1)

        # Calculate the gradients along the yi axis
        dzi_dy = np.gradient(zi, yi[:, 0], axis=0)

        # Calculate the second derivatives along the xi axis
        d2zi_dxi2 = np.gradient(dzi_dx, xi[0], axis=1)

        # Compute the local volatility using the discretized Dupire formula
        #local_volatility = np.sqrt(2 * zi / (xi[0] ** 2 * d2zi_dxi2)
        #                          - (dzi_dx ** 2) / xi[0] * zi)

        local_volatility = np.sqrt((zi ** 2 + 2 * zi * yi * (dzi_dy + (r - self.get_div()) * xi * dzi_dx)) /
                                   ((1 + xi * self.get_div() * np.sqrt(yi)) ** 2 + zi * xi ** 2 * yi *
                                    (d2zi_dxi2 - self.get_div() * xi ** 2 * yi)))
        return local_volatility

    def build_local_vol_surface_v2(self, r, xi, yi, zi):
        # Compute dzi_dx using central differences
        dzi_dx = np.zeros_like(zi)
        dzi_dx[:, 1:-1] = (zi[:, 2:] - zi[:, :-2]) / (xi[0, 2:] - xi[0, :-2])

        # Compute dzi_dy using central differences
        dzi_dy = np.zeros_like(zi)
        dzi_dy[1:-1, :] = (zi[2:, :] - zi[:-2, :]) / (yi[2:, np.newaxis] - yi[:-2, np.newaxis])

        # Compute d2zi_dxi2 using central differences
        d2zi_dxi2 = np.zeros_like(zi)
        d2zi_dxi2[:, 1:-1] = (zi[:, 2:] - 2 * zi[:, 1:-1] + zi[:, :-2]) / (xi[0, 2:] - xi[0, :-2]) ** 2

        # Apply boundary conditions for dzi_dx and d2zi_dxi2
        dzi_dx[:, 0] = (zi[:, 1] - zi[:, 0]) / (xi[0, 1] - xi[0, 0])  # Forward difference at boundary
        dzi_dx[:, -1] = (zi[:, -1] - zi[:, -2]) / (xi[0, -1] - xi[0, -2])  # Backward difference at boundary
        d2zi_dxi2[:, 0] = (zi[:, 2] - 2 * zi[:, 1] + zi[:, 0]) / (
                    xi[0, 2] - xi[0, 0]) ** 2  # Forward difference at boundary
        d2zi_dxi2[:, -1] = (zi[:, -3] - 2 * zi[:, -2] + zi[:, -1]) / (
                    xi[0, -3] - xi[0, -1]) ** 2  # Backward difference at boundary

        local_volatility = np.sqrt((zi ** 2 + 2 * zi * yi * (dzi_dy + (r - self.get_div()) * xi * dzi_dx)) /
                                   ((1 + xi * self.get_div() * np.sqrt(yi)) ** 2 + zi * xi ** 2 * yi *
                                    (d2zi_dxi2 - self.get_div() * xi ** 2 * yi)))
        return local_volatility
