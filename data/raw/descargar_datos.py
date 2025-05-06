import yfinance as yf
import pandas as pd
import os

etfs = ["SPY", "GLD", "TLT", "HYG", "VNQ", "VB"]

raw_path = "data/raw"
os.makedirs(raw_path, exist_ok=True)

for etf in etfs:
    print(f"Descargando datos para {etf}...")
    
    # Descarga de datos desde el 2007 sin multi-nivel
    data = yf.download(etf, start="2007-01-01", end="2025-12-31", interval="1d", auto_adjust=False)

    # Nos aseguramos de que el índice sea una columna normal
    data.reset_index(inplace=True)

    # Guardamos en CSV
    file_path = os.path.join(raw_path, f"{etf}.csv")
    data.to_csv(file_path, index=False)

    print(f"Datos para {etf} guardados en {file_path}")

print("Todos los datos han sido descargados correctamente.")

