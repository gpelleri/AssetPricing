from assetpricing.products.security import *
from ...utils.types import *
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
        df = yf.download(ticker)
        self.price = df['Close']

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

    def __SetVol__(self, volatility):
        """
        Private setter : must ONLY be called by derivatives class when extracting implied vol
        :param volatility:
        """
        self._vol = volatility

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

        return chains
