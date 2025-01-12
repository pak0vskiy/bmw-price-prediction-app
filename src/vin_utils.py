import requests
import pandas as pd
from time import sleep

def get_vin_data(vin: str, url: str = "https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVinValues/") -> dict:
    vin = vin.strip().upper()
    r = requests.get(f"{url}{vin}?format=json", timeout=10)
    if r.status_code == 200:
        return r.json()  # Parse JSON response
        
    else:
        print(f"Error: {r.status_code}, {r.text}")
        return {}