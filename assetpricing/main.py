import numpy as np
import datetime as dt
import pandas as pd


def generate_spots(size=(3000, 50), volatility=0.20, initial_level=100.):
    random_normal = np.random.normal(0, 1, size)
    random_lognormal = np.exp(volatility * np.sqrt(1 / 250) * random_normal)
    random_lognormal[0] = initial_level
    return np.cumprod(random_lognormal, axis=0)


nb_stocks = 1500
list_letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U",
                "V", "W", "X", "Y", "Z"]
list_permut = np.random.choice(list_letters, size=(nb_stocks, 5), replace=True)
list_tickers = sorted(set([f"{''.join(tick)} XX Equity" for tick in list_permut]))
list_dates = [dt.datetime(year, month, day) for year in range(2000, 2022) for day, month in
              ((1, 1), (1, 4), (1, 7), (1, 10))]
list_currencies = ["EUR", "USD", "CAD", "JPY", "GBP", "CHF", "CZK"]
nb_dates = len(list_dates)

df_composition = pd.DataFrame(
    {ticker: pd.Series(np.random.choice([0, 1], size=nb_dates, replace=True), index=list_dates) for ticker in
     list_tickers})
series_sectors = pd.Series(np.random.choice(
    ["Utilities", "Financials", "Industrials", "Energy", "Real Estate", "Health Care", "Consumer Staples",
     "Consumer Discretionary", "Communication Services", "Materials", "Information Technology"], size=nb_stocks,
    replace=True), index=list_tickers)
series_currencies = pd.Series(
    np.random.choice(list_currencies, size=nb_stocks, replace=True, p=[0.25, 0.50, 0.10, 0.05, 0.03, 0.06, 0.01]),
    index=list_tickers)
df_esg_scores = pd.DataFrame(np.random.uniform(0, 1, (nb_dates, nb_stocks)) * 10, index=list_dates,
                             columns=list_tickers)
df_market_caps = pd.DataFrame(np.random.uniform(0, 1, (nb_dates, nb_stocks)) * 100000, index=list_dates,
                              columns=list_tickers)
df_fx = pd.DataFrame(generate_spots(size=(nb_dates, len(list_currencies) - 1), volatility=0.10, initial_level=1.),
                     index=list_dates, columns=[f"{cur}EUR Curncy" for cur in list_currencies if cur != "EUR"])

df_price = pd.DataFrame(generate_spots(df_market_caps.shape), index=list_dates, columns=list_tickers)


class IndexSelection():

    def __init__(self, df_composition, series_sectors, series_currencies, df_esg_scores, df_market_caps, df_fx):
        self.df_composition = df_composition
        self.series_sectors = series_sectors
        self.series_currencies = series_currencies
        self.df_esg_scores = df_esg_scores
        self.df_market_caps = df_market_caps
        self.df_fx = df_fx
        # Will be used to store the mkt_cap converted to euros
        self.converted_mkt_cap = None

        # I took the initiative to change parameters name to avoid shadowing, since we have defined the variables in
        # a global context

    def compute_market_cap_selection(self, df_compo, df_mkt, currencies):
        # Bonus 1 : conversion
        df = self.df_fx.rename(columns=lambda x: x[:3])
        df['EUR'] = 1

        # this isn't probably the most efficient way to achieve it since i'm iterating with a loop
        def convert(row):
            date = row.name
            for equity, cap in row.iteritems():
                symbol = currencies.loc[equity]
                row[equity] = cap * df.loc[date, symbol]
            return row

        converted_mc = df_mkt.apply(convert, axis=1)
        # Stores it for bonus question
        self.converted_mkt_cap = converted_mc

        mkt_cap_filter = converted_mc >= 2000
        init_compo_filter = df_compo > 0
        # combine filters as they're boolean
        return mkt_cap_filter & init_compo_filter

    def compute_sector_selection(self, df, sectors):
        sector_filter = sectors != 'Financials'
        # combine filters as they're boolean
        return df & sector_filter

    def compute_esg_selection(self, df, df_esg):
        # filter df_esg, from boolean to raw numbers
        a = df_esg.where(df, other=-float("inf"))
        top_50_values = a.apply(lambda row: sorted(row[row != -np.inf], reverse=True)[:50], axis=1)
        # Back to boolean dataframe
        df_bool = a.apply(lambda row: row.isin(top_50_values.loc[row.name]), axis=1).astype(int)
        return df_bool

    # N.B : My initial strategy was to add an equal weight to each stock below 0.1 but if a stock if 0.099 ,
    # reallocating a weight could make it exceed 0.1. I still didn't manage to see a case where weights are =100%
    # and one of the initial weights is above 10% so i might have missed something while doing my own testing
    def compute_weighting_scheme(self, df, mkt_cap):
        # Apply boolean mask
        mkt_cap = mkt_cap[df.astype(bool)]
        # Compute market cap weights for each date
        total = mkt_cap.sum(axis=1)
        weights = mkt_cap.div(total, axis=0)

        weights[weights > 0.1] = 0.1
        excess = 1 - weights.sum(axis=1)

        # Bad "convergence" so i'm setting some kind of tolerance
        while excess.round(6).any() > 0.000001:
            # Find stocks that exceed the 0.1 weight limit
            stocks_below_max = (weights < 0.1).sum(axis=1)

            # Reallocate excess weight to stocks below the 0.1 weight limit
            for i, (excess, nb) in enumerate(zip(excess, stocks_below_max)):
                if excess > 0 and nb > 0:
                    excess_per_stock = excess / nb
                    weights.iloc[i] += excess_per_stock

            # in case rebalancing was too much, reset to max 0.1, compute weights and goes back to the while instruction
            weights[weights > 0.1] = 0.1
            excess = 1 - weights.sum(axis=1)

        # back to normal question
        weights *= 100
        # dvlp safety check
        # total_weights = weights.sum(axis=1)
        # weights_max = weights.max(axis=1)
        return weights

    def run(self):
        # We will compute each filter and pass the return to the next filter as a param
        mkt_cap = self.compute_market_cap_selection(self.df_composition, self.df_market_caps, self.series_currencies)
        sector = self.compute_sector_selection(mkt_cap, self.series_sectors)
        esg = self.compute_esg_selection(sector, self.df_esg_scores)
        # Pass the converted market cap as param since we have it now
        weights = self.compute_weighting_scheme(esg, self.converted_mkt_cap)

        return weights

    def run_bonus(self):
        mkt_cap = self.compute_market_cap_selection(self.df_composition, self.df_market_caps, self.series_currencies)
        sector = self.compute_sector_selection(mkt_cap, self.series_sectors)
        esg = self.compute_esg_selection(sector, self.df_esg_scores)
        dico = self.dictionary_bonus(self.df_composition, mkt_cap, sector, esg)
        return dico

    def dictionary_bonus(self, df_init_compo, caps_filter, sector_filter, esg_filter):
        def stock_filter(date, initial, caps, sectors, esg):
            df = pd.DataFrame(index=caps.index)
            df['Initial Composition Filter'] = initial
            df['Market cap Filter'] = caps
            df['Sector Filter'] = sectors
            df['ESG Filter'] = esg
            # additional columns with data used for filtering
            df['Market Cap'] = self.converted_mkt_cap.loc[date]
            df['Sector'] = self.series_sectors
            df['ESG Score'] = self.df_esg_scores.loc[date]

            return df

        my_dict = {date: stock_filter(date, df_init_compo.astype(bool).loc[date], caps_filter.loc[date],
                                      sector_filter.loc[date], esg_filter.loc[date]) for date in df_init_compo.index}
        return my_dict

    # Iterating version, terrible performances
    # def bonus_bad(self, df_init_compo, caps_filter, sector_filter, esg_filter):
    #
    #     def stock_filter(date, initial, caps, sectors, esg):
    #         df = pd.DataFrame(index=caps.columns)
    #         for col in caps.columns:
    #             # booleans columns
    #             df.loc[col, 'Initial Composition Filter'] = initial.loc[date, col]
    #             df.loc[col, 'Market cap Filter'] = caps.loc[date, col]
    #             df.loc[col, 'Sector Filter'] = sectors.loc[date, col]
    #             df.loc[col, 'ESG Filter'] = esg.loc[date, col]
    #             # additional columns with data used for filtering
    #             df.loc[col, 'Market Cap'] = self.converted_mkt_cap.loc[date, col]
    #             df.loc[col, 'Sector'] = self.series_sectors.loc[col]
    #             df.loc[col, 'ESG Score'] = self.df_esg_scores.loc[date, col]
    #         return df
    #
    #     my_dict = {date: stock_filter(date, df_init_compo.astype(bool), caps_filter, sector_filter, esg_filter) for date
    #                in
    #                df_init_compo.index}
    #     return my_dict


if __name__ == '__main__':
    index = IndexSelection(df_composition, series_sectors, series_currencies, df_esg_scores, df_market_caps, df_fx)
    # my_weights = index.run()
    my_dico = index.run_bonus()
    print(my_dico)
