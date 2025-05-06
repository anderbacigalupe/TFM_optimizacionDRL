import pandas as pd
import numpy as np
from pathlib import Path

def calculate_log_returns():
    # Definir rutas relativas desde la ubicación del script en utils
    base_dir = Path(__file__).parent.parent
    processed_path = base_dir / "data" / "processed"
    
    # Rutas de archivos
    input_path = processed_path / "processed_prices.csv"
    output_path = processed_path / "etf_log_returns.csv"
    
    print(f"Leyendo datos de precios desde: {input_path}")
    
    # Leer el archivo CSV con los precios ajustados
    try:
        prices_df = pd.read_csv(input_path)
        prices_df['Date'] = pd.to_datetime(prices_df['Date'])
        prices_df = prices_df.set_index('Date')
    except Exception as e:
        print(f"Error al leer el archivo de precios: {e}")
        return
    
    print(f"Calculando retornos logarítmicos para {len(prices_df.columns)} ETFs")
    
    # Calcular los retornos logarítmicos: ln(precio_hoy / precio_ayer)
    log_returns_df = np.log(prices_df / prices_df.shift(1))
    
    # La primera fila tendrá valores NaN porque no hay precio anterior
    # Eliminar filas con todos NaN
    log_returns_df = log_returns_df.dropna(how='all')
    
    # Resetear el índice para volver a tener 'Date' como columna
    log_returns_df = log_returns_df.reset_index()
    
    # Guardar los retornos logarítmicos en un nuevo CSV
    log_returns_df.to_csv(output_path, index=False)
    
    print(f"\nProceso completado! Retornos logarítmicos guardados en: {output_path}")
    print(f"Periodo de tiempo: {log_returns_df['Date'].min()} a {log_returns_df['Date'].max()}")
    print(f"Número de observaciones: {len(log_returns_df)}")
    
    # Mostrar estadísticas descriptivas de los retornos
    stats_df = log_returns_df.drop('Date', axis=1).describe().T
    stats_df['annualized_volatility'] = stats_df['std'] * np.sqrt(252)  # Aprox. 252 días de trading al año
    stats_df['annualized_return'] = stats_df['mean'] * 252
    
    print("\nEstadísticas descriptivas de los retornos logarítmicos diarios:")
    print(stats_df[['mean', 'std', 'annualized_return', 'annualized_volatility']])

if __name__ == "__main__":
    calculate_log_returns()