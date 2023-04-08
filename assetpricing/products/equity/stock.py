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
            self.downloadPrice(name)  # set price to current spot price

        self.div = div
        self._derivatives = None

    def getName(self):
        return self.name

    def getVol(self):
        return self._vol

    def getDiv(self):
        return self.div

    def getDerivatives(self):
        return self._derivatives

    # TODO probably poor naming
    # What if we want to extract past prices and not only spot ? -> out of scope atm
    def downloadPrice(self, ticker):
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

    def __SetVol__(self, vol):
        """
        Private setter : must ONLY be called by derivatives class when extracting implied vol
        """
        self._vol = vol

    def getOptionData(self):
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


    def clean_Option_Data(self, df: DataFrame):
        """
        This function is a must called before creating a vol surface. It cleans dataset removing irrelevant data
        based on the following conditions :
        - trade must have occured within last business days
        - spread must be smaller than expiry spread min + 4 * expiry spread std.
        - volume / open interest has to be valid
        - and obviously we remove out of the money data (Arbitrary conditions for now)
        """

        # Filter out options that haven't been traded in the last 5 business days
        df['lastTradeDate'] = pd.to_datetime(df['lastTradeDate'])
        five_days_ago = pd.Timestamp.now().normalize() - pd.offsets.BDay(5)
        five_days_ago = five_days_ago.tz_localize('UTC')
        df = df[df['lastTradeDate'] > five_days_ago]

        # # Filter out options with high bid/ask spreads
        df['spread'] = df['ask'] - df['bid']
        spread_mean = df.groupby('Expiry')['spread'].mean()
        spread_std = df.groupby('Expiry')['spread'].std()
        merged = pd.concat([spread_mean, spread_std], axis=1).reset_index()
        merged.columns = ['Expiry', 'mean', 'std']
        df = df.merge(merged, on='Expiry')
        df = df[df['spread'] < (df['mean'] + 4 * df['std'])]

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


    def get_Skew(self, T, df: DataFrame):
        # filter the input DataFrame to only contain data for the given `Expiry`
        df_filtered = df[df['Expiry'] == T]

        # filter the DF. We only use put data for now, for which implied vol is sufficient
        puts = df_filtered[(df_filtered['impliedVolatility'] > 0.0001)].copy()

        puts['imp'] = puts.apply(implied_volatility_row, axis=1)

        return puts

    def interpolate_Skew(self, df: DataFrame, method='linear'):

        # filter the DataFrame to only contain 'Expiry' between 0.5 and 2.5
        expiry_filtered = df[(df['Expiry'] >= 0.5) & (df['Expiry'] <= 2.5)]

        # group the filtered DataFrame by 'Expiry' and apply the `get_Skew` function to each group
        expiries = expiry_filtered['Expiry'].unique()
        interpolations = []
        for T in expiries:
            skew_df = self.get_Skew(T, expiry_filtered)
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

    def build_Impl_Vol_Surface(self, df: DataFrame, option_type: OptionTypes, method='linear'):
        options_filtered = df[df['OptionType'] == option_type.value]

        interp_skews = self.interpolate_Skew(options_filtered, method)

        x = interp_skews['strike'].values
        y = interp_skews['Expiry'].values
        z = interp_skews['interpolation'].values

        # define RBF parameters
        rbf_type = 'linear'
        epsilon = 2.0
        rbf = Rbf(x, y, z, function=rbf_type, epsilon=epsilon)

        # create a grid of x and y values for the plot
        xi = np.linspace(min(x), max(x), 100)
        yi = np.linspace(min(y), max(y), 100)
        xi, yi = np.meshgrid(xi, yi)

        # evaluate interpolated surface using RBF
        zi = rbf(xi, yi)

        return xi.flatten(), yi.flatten(), zi.flatten()

    def plot_Vol_Surface(self, xi, yi, zi):

        x = xi.reshape(100,100)
        y = yi.reshape(100,100)
        z = zi.reshape(100,100)

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