import pandas as pd
import os
from pathlib import Path

def combine_etf_data():
    # Definir rutas relativas desde la ubicación del script en utils
    base_dir = Path(__file__).parent.parent
    raw_path = base_dir / "data" / "raw"
    processed_path = base_dir / "data" / "processed"
    
    # Asegurar que la carpeta processed existe
    processed_path.mkdir(parents=True, exist_ok=True)
    
    # Diccionario para almacenar los dataframes por ETF
    etf_dfs = {}
    
    # Obtener todos los archivos CSV en la carpeta raw
    csv_files = list(raw_path.glob("*.csv"))
    
    print(f"Encontrados {len(csv_files)} archivos CSV en la carpeta raw.")
    
    for csv_file in csv_files:
        # Extraer el nombre del ETF del nombre del archivo (sin extensión)
        etf_name = csv_file.stem
        
        print(f"Procesando el archivo: {csv_file.name}")
        
        try:
            # Leer el CSV, saltando la primera fila que tiene encabezados problemáticos
            df = pd.read_csv(csv_file, skiprows=1)
            
            # Si la primera columna no se llama 'Date', ajustar manualmente las columnas
            if df.columns[0] != 'Date':
                columns = ['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
                df.columns = columns
            
            # Seleccionar solo las columnas 'Date' y 'Adj Close'
            df = df[['Date', 'Adj Close']]
            
            # Asegurarse de que 'Date' está en formato de fecha
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Renombrar 'Adj Close' para identificar el ETF
            df = df.rename(columns={'Adj Close': etf_name})
            
            # Usar la fecha como índice
            df = df.set_index('Date')
            
            # Guardar en el diccionario
            etf_dfs[etf_name] = df
            
            print(f"  - Datos cargados correctamente: {len(df)} filas para {etf_name}")
            
        except Exception as e:
            print(f"Error al procesar {csv_file.name}: {e}")
    
    if not etf_dfs:
        print("No se pudieron cargar datos de ningún archivo CSV.")
        return
    
    # Combinar todos los dataframes usando join para alinear las fechas
    combined_df = pd.concat(etf_dfs.values(), axis=1)
    
    # Resetear el índice para tener 'Date' como columna
    combined_df = combined_df.reset_index()
    
    # Ordenar por fecha
    combined_df = combined_df.sort_values('Date')
    
    # Guardar el dataframe combinado
    output_path = processed_path / "processed_prices.csv"
    combined_df.to_csv(output_path, index=False)
    
    print(f"\nProceso completado! Datos combinados guardados en: {output_path}")
    print(f"Total de filas en el archivo combinado: {len(combined_df)}")
    print(f"ETFs incluidos: {', '.join(etf_dfs.keys())}")

if __name__ == "__main__":
    combine_etf_data()