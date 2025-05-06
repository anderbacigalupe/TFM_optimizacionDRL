import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from entorno.entorno_cartera import PortfolioEnv  # Asegúrate de importar la clase correctamente
import os

# Cargar los datos procesados (supongo que ya los tienes listos en 'processed_prices.csv')
data = pd.read_csv('data/processed/processed_prices.csv', index_col='Date', parse_dates=True)

# Asegurarse de que los datos estén alineados con las fechas correctas
data = data.loc['2007-04-11':'2025-04-10']  # Filtramos las fechas de los datos

# Definir los ETFs que forman parte de la cartera
etfs = ['GLD', 'HYG', 'SPY', 'TLT', 'VB', 'VNQ']

# Calcular los retornos logarítmicos diarios
log_returns = np.log(data[etfs] / data[etfs].shift(1))

# Inicializar la clase portfolio.env
env = PortfolioEnv(data[etfs].values)

# Inicializar variables para la simulación
initial_balance = 1000000  # Balance inicial en USD
n_periods = len(data)  # Número de periodos en los datos

# Simulación de cartera equiponderada
weights = np.ones(len(etfs)) / len(etfs)  # Peso igual para cada ETF

# Almacenar el valor de la cartera a lo largo del tiempo
portfolio_values = np.zeros(n_periods)
portfolio_values[0] = initial_balance

# Realizar la simulación mes a mes (rebalanceo mensual)
for t in range(1, n_periods):
    portfolio_values[t] = portfolio_values[t-1] * (1 + np.dot(log_returns.iloc[t], weights))

# Visualizar los resultados
plt.figure(figsize=(10, 6))
plt.plot(data.index, portfolio_values, label='Cartera Equiponderada', color='blue')
plt.title('Simulación de Cartera Equiponderada de 6 ETFs')
plt.xlabel('Fecha')
plt.ylabel('Valor de la Cartera (USD)')
plt.legend()
plt.grid(True)
plt.show()

# Guardar el gráfico en la carpeta 'figures'
plt.savefig('figures/simulacion_equiponderada.png')

# 1. Rendimiento anualizado
daily_return = log_returns.mean().mean()  # Promedio de los retornos diarios de todos los ETFs
annualized_return = (1 + daily_return) ** 252 - 1  # Aproximación usando 252 días hábiles al año

# 2. Volatilidad anualizada
daily_volatility = log_returns.std().mean()  # Promedio de la volatilidad diaria de todos los ETFs
annualized_volatility = daily_volatility * np.sqrt(252)

# 3. Ratio Sharpe (sin tasa libre de riesgo, asumimos 0%)
sharpe_ratio = annualized_return / annualized_volatility

# 4. Máximo Drawdown
cum_returns = np.cumprod(1 + log_returns, axis=0)  # Rentabilidad acumulada
peak = np.maximum.accumulate(cum_returns)  # Picos anteriores
drawdowns = (cum_returns - peak) / peak  # Drawdown
max_drawdown = drawdowns.min().min()  # El máximo drawdown es el mínimo valor de drawdowns

# 5. VaR95 (anualizado)
daily_var = log_returns.quantile(0.05).mean()  # Percentil 5 (VaR95)
annualized_var = daily_var * np.sqrt(252)

# Mostrar las métricas
print(f"Rendimiento Anualizado: {annualized_return * 100:.2f}%")
print(f"Volatilidad Anualizada: {annualized_volatility * 100:.2f}%")
print(f"Ratio Sharpe: {sharpe_ratio:.2f}")
print(f"Máximo Drawdown: {max_drawdown * 100:.2f}%")
print(f"VaR95: {annualized_var * 100:.2f}%")


# Guardar los resultados en un archivo CSV (opcional)
results = pd.DataFrame({'Fecha': data.index, 'Valor de la Cartera': portfolio_values})
results.to_csv('results/simulacion_equiponderada_resultados.csv', index=False)