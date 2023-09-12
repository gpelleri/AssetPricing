import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
from numpy import meshgrid

from assetpricing.products.equity import Stock
from assetpricing.utils.vol_utils import *


class LocalVol:

    def __init__(self):
        self._vol_matrix = None
        self._model = None
        return

    def get_vol_matrix(self):
        return self._vol_matrix

    def get_skew(self, T, df: DataFrame):
        """
        Extract Implied volatility from options prices for a given maturity T
        """
        # filter the input DataFrame to only contain data for the given `Expiry`
        df_filtered = df[df['Expiry'] == T]

        # filter the DF. We only use put data for now, for which implied vol is sufficient
        puts = df_filtered[(df_filtered['impliedVolatility'] > 0.0001)].copy()

        puts['impliedVol'] = puts.apply(implied_volatility_row, axis=1)
        puts.set_index('strike', inplace=True)
        return puts['impliedVol']

    def implied_vol_surface(self, df):
        """
        Computes implied volatility interpolation based on implied volatility extracted from options prices
        """

        # filter the DataFrame to only contain 'Expiry' between 0.5 and 2.5
        expiry_filtered = df[(df['Expiry'] >= 0.5) & (df['Expiry'] <= 2.5)]

        # group the filtered DataFrame by 'Expiry' and apply the `get_Skew` function to each group
        expiries = expiry_filtered['Expiry'].unique()
        vol_matrix = {}
        for T in expiries:
            skew_df = self.get_skew(T, expiry_filtered)
            vol_matrix[T] = skew_df

        vol_matrix = pd.DataFrame(vol_matrix)

        self._vol_matrix = vol_matrix
        return

    def plot_implied_volatility_surface(self):
        df = self.get_vol_matrix()
        expiry_values = df.columns
        strike_values = df.index
        Y, X = meshgrid(expiry_values, strike_values)
        Z = df.values

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Use the 'hot' colormap to represent higher Z values as redder
        surf = ax.plot_surface(X, Y, Z, cmap="coolwarm")

        # Add a color bar to the plot
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.2)

        ax.set_xlabel('Strike')
        ax.set_ylabel('Expiry')
        ax.set_zlabel('Implied Volatility')
        ax.set_title('Implied Volatility Surface')

        plt.show()

    def fit_model(self):
        vol_surface = vol.get_vol_matrix()
        # Create lists to store the data for all combinations of strikes and expiries
        strike_values = []
        expiry_values = []
        implied_volatility_values = []

        # Iterate through all combinations of expiry and strike and collect the data
        for expiry in vol_surface.columns:
            for strike in vol_surface.index:
                strike_values.append(strike)
                expiry_values.append(expiry)
                implied_volatility_values.append(vol_surface.loc[strike, expiry])

        # Create a DataFrame to store the data
        data = pd.DataFrame({
            'Strike': strike_values,
            'Expiry': expiry_values,
            'ImpliedVolatility': implied_volatility_values,
            'K': [strike ** 1 for strike in strike_values],
            'T': [expiry for expiry in expiry_values],
            'K^2': [strike ** 2 for strike in strike_values],
            'T^2': [expiry ** 2 for expiry in expiry_values],
            'T*K': [expiry * strike for strike, expiry in zip(strike_values, expiry_values)],
            'Constant': 1  # Add a constant term
        })

        # remove all np.nan in implied vol
        data = data.dropna()
        # Fit the OLS regression model
        model = sm.OLS(data['ImpliedVolatility'], data[['K', 'T', 'K^2', 'T^2', 'T*K', 'Constant']]).fit()

        self._set_model(model)

        return

    def _set_model(self, model):
        self._model = model

    def get_model(self):
        return self._model

    def compute_local_vol(self, spot, strike, expiry):
        model = self.get_model()
        desired_K = strike  # Replace with your desired value of K
        desired_T = expiry  # Replace with your desired value of T

        # Calculate K², T², and T*K
        desired_K2 = desired_K ** 2
        desired_T2 = desired_T ** 2
        desired_TK = desired_K * desired_T

        # Create a DataFrame with the desired values of K, T, K², T², and T*K
        input_data = pd.DataFrame({
            'K': [desired_K],
            'T': [desired_T],
            'K^2': [desired_K2],
            'T^2': [desired_T2],
            'T*K': [desired_TK],
            'Constant': [1]
        })

        # Use the trained model to predict implied volatility
        implied_variance = model.predict(input_data[['K', 'T', 'K^2', 'T^2', 'T*K', 'Constant']])

        return implied_variance

    def compute_local_vol_surface(self, dico):
        "WIP"
        return

if __name__ == '__main__':
    r = 0.0542  # 1 year risk-free risk in %
    q = 0.0054  # apple annual dividend rate in %

    apple = Stock("AAPL", False)
    chains = apple.get_option_data()
    chains['Dividend'] = q
    chains['Spot'] = apple.get_price()
    chains['Risk-Free Rate'] = r

    cleaned = apple.clean_option_data(chains)
    vol = LocalVol()
    cleaned = cleaned[cleaned['OptionType'] == 2]

    vol.implied_vol_surface(cleaned)
    vol.plot_implied_volatility_surface()
    # Call the function with your DataFrame
    # vol.plot_implied_volatility_surface()
    vol.fit_model()
    print(vol.get_model().summary())

    print(vol.compute_local_vol(179, 175, 0.01069))

