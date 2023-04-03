import os
import pandas as pd


def read_csv_file(filename):
    """
    Reads a CSV file from the parent directory of the current working directory
    and creates a dataframe with the data.

    Parameters:
        filename (str): The name of the CSV file to read.

    Returns:
        pandas.DataFrame: A dataframe containing the data from the CSV file.
    """
    # Get the path to the parent directory of the current working directory
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

    # Construct the full path to the CSV file
    filepath = os.path.join(parent_dir, filename)

    # Use pandas to read the CSV file from the given filepath
    df = pd.read_csv(filepath)

    # Return the dataframe
    return df


df = read_csv_file("options_data.csv")



calls = chains[chains["optionType"] == "put"]

# print the expirations
set(calls.expiration)
#
# # select an expiration to plot
calls_at_expiry = calls[calls["expiration"] == "2023-05-05 23:59:59"]
#
# # filter out low vols
filtered_calls_at_expiry = calls_at_expiry[calls_at_expiry.impliedVolatility >= 0.0001]
#
# # set the strike as the index so pandas plots nicely
filtered_calls_at_expiry[["strike", "impliedVolatility"]].set_index("strike").plot(
     title="Implied Volatility Skew", figsize=(7, 4)
)