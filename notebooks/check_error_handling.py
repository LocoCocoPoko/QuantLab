from src.quantlab.data.fetch import fetch_price_data

try:
    fetch_price_data(["INVALIDTICKERXYZ"], "2021-01-01", "2024-01-01")
except ValueError as e:
    print(f"Caught expected error: {e}")