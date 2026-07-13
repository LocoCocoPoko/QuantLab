from src.quantlab.data.fetch import fetch_price_data

prices = fetch_price_data(
    tickers = ["AAPL", "MSFT", "GOOG"],
    start_date = "2023-01-01",
    end_date = "2023-06-01"
)

print(prices.head()) # Shows first 4 rows
print(prices.shape) # Returns (rows,columns) as a tuple. How many trading days we get and did we get exactly 3 columns(one per ticker - Apple, Google, Microsoft)?
print(prices.dtypes) # Returns the data types of each column. 
print(prices.isna().sum()) # Returns the number of missing values in each column. If we have any missing values, we need to investigate why and how to handle them.
