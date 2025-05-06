import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os
from datetime import datetime, timedelta
import calendar

# Crear carpeta results si no existe
if not os.path.exists('results'):
    os.makedirs('results')

# Cargar los datos
df = pd.read_csv('data/processed/processed_prices.csv', index_col='Date', parse_dates=True)

# Función para calcular los retornos
def calculate_returns(prices):
    return prices.pct_change().dropna()

# Función para calcular el ratio Sharpe
def calculate_sharpe_ratio(weights, returns, risk_free_rate=0.0):
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return (portfolio_return - risk_free_rate) / portfolio_stddev

# Función para optimizar el ratio Sharpe (encontrando el negativo para maximizar)
def negative_sharpe_ratio(weights, returns, risk_free_rate=0.0):
    return -calculate_sharpe_ratio(weights, returns, risk_free_rate)

# Función para obtener los pesos óptimos con restricción de peso mínimo
def get_optimal_weights(returns, risk_free_rate=0.0, min_weight=0.05):
    n_assets = len(returns.columns)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((min_weight, 1) for _ in range(n_assets))  # Peso mínimo de 5%
    
    # Verificar si la restricción de peso mínimo es factible
    if n_assets * min_weight > 1:
        # Si no es factible, usar equiponderado
        return np.array([1/n_assets] * n_assets)
    
    initial_weights = np.array([1/n_assets] * n_assets)
    
    result = minimize(
        negative_sharpe_ratio,
        initial_weights,
        args=(returns, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    # Si la optimización no converge, usar pesos equiponderados
    if not result.success:
        return np.array([1/n_assets] * n_assets)
    
    return result['x']

# Función para calcular el número de participaciones enteras a comprar
def calculate_shares(weights, prices, capital):
    prices_vector = prices.values
    shares_float = weights * capital / prices_vector
    shares = np.floor(shares_float)  # Redondeamos hacia abajo para obtener números enteros
    
    return shares

# Función para calcular costes de transacción
def calculate_transaction_costs(old_shares, new_shares, prices):
    # Calculamos la diferencia absoluta de participaciones
    shares_diff = np.abs(new_shares - old_shares)
    
    # Calculamos el coste por participación con un mínimo de 0.35 USD por operación
    operation_costs = np.maximum(shares_diff * 0.0035, np.where(shares_diff > 0, 0.35, 0))
    
    # Aplicamos el tope del 1% del valor de la transacción
    transaction_values = shares_diff * prices.values
    max_cost = transaction_values * 0.01
    operation_costs = np.minimum(operation_costs, max_cost)
    
    # Sumamos el deslizamiento del 0.1%
    slippage_costs = transaction_values * 0.001
    
    return operation_costs.sum() + slippage_costs.sum()

# Función para calcular métricas de rendimiento
def calculate_metrics(returns):
    # Rentabilidad anualizada
    ann_return = returns.mean() * 252
    
    # Volatilidad anualizada
    ann_volatility = returns.std() * np.sqrt(252)
    
    # Ratio Sharpe (asumiendo risk-free rate = 0)
    sharpe_ratio = ann_return / ann_volatility
    
    # Ratio Sortino (usando solo rendimientos negativos)
    negative_returns = returns.copy()
    negative_returns[negative_returns > 0] = 0
    ann_downside_volatility = negative_returns.std() * np.sqrt(252)
    sortino_ratio = ann_return / ann_downside_volatility if ann_downside_volatility != 0 else 0
    
    # VaR 95%
    var_95 = np.percentile(returns, 5)
    
    # Máximo drawdown
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns / running_max - 1) * 100  # En porcentaje
    max_drawdown = drawdown.min()
    
    return {
        'Ann. Return (%)': ann_return * 100,
        'Ann. Volatility (%)': ann_volatility * 100,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'VaR 95%': var_95 * 100,
        'Max Drawdown (%)': max_drawdown 
    }

# Función para ajustar pesos para asegurar que cumplan con el mínimo
def adjust_weights_for_minimum(weights, min_weight=0.05):
    n_assets = len(weights)
    adjusted_weights = np.maximum(weights, min_weight)
    
    # Normalizar para que sumen 1
    adjusted_weights = adjusted_weights / adjusted_weights.sum()
    
    return adjusted_weights

# Función principal para ejecutar la simulación
def run_simulation(prices_df, initial_capital=1000000, risk_free_rate=0.0, min_weight=0.05):
    # Inicializar variables
    portfolio_values = {}
    all_returns = []
    metrics_history = []
    current_shares = None
    remaining_cash = initial_capital
    
    # Convertir el índice a datetime si no lo es ya
    if not isinstance(prices_df.index, pd.DatetimeIndex):
        prices_df.index = pd.to_datetime(prices_df.index)
    
    # Obtener la fecha de inicio y fin
    start_date = datetime(2007, 4, 11)
    end_date = datetime(2025, 4, 10)
    
    # Filtrar datos por fecha
    prices_df = prices_df.loc[(prices_df.index >= start_date) & (prices_df.index <= end_date)]
    
    # Generar meses para el walkforward
    months = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    # Inicializar con cartera equiponderada el primer mes
    first_month_prices = prices_df.loc[(prices_df.index >= start_date) & (prices_df.index < months[1])]
    first_day_prices = first_month_prices.iloc[0]
    
    n_assets = len(first_day_prices)
    equal_weights = np.array([1/n_assets] * n_assets)
    
    # Calcular participaciones iniciales
    current_shares = calculate_shares(equal_weights, first_day_prices, initial_capital)
    initial_investment = (current_shares * first_day_prices).sum()
    remaining_cash = initial_capital - initial_investment
    
    # Registrar valor inicial de la cartera
    portfolio_values[start_date] = initial_investment + remaining_cash
    
    # Guardar pesos históricos para análisis
    weights_history = {start_date: equal_weights}
    
    # Para cada mes en el rango de fechas
    for i in range(1, len(months)):
        if months[i] > end_date:
            break
            
        # Mes actual para aplicar la estrategia
        current_month_start = months[i]
        
        # Calcular el final del mes actual
        if i < len(months) - 1:
            next_month_start = months[i+1]
        else:
            # Para el último mes, usar la fecha de fin
            next_month_start = end_date + timedelta(days=1)
        
        # Datos históricos hasta el mes actual (para optimización)
        historical_prices = prices_df.loc[prices_df.index < current_month_start]
        
        if not historical_prices.empty:
            historical_returns = calculate_returns(historical_prices)
            
            # Optimizar los pesos basados en datos históricos con restricción de peso mínimo
            optimal_weights = get_optimal_weights(historical_returns, risk_free_rate, min_weight)
            
            # Asegurar que todos los pesos cumplan con el mínimo
            optimal_weights = adjust_weights_for_minimum(optimal_weights, min_weight)
            
            # Guardar los pesos para análisis
            weights_history[current_month_start] = optimal_weights
            
            # Precios del primer día del mes actual
            current_day_prices = prices_df.loc[prices_df.index >= current_month_start].iloc[0]
            
            # Valorar la cartera actual
            current_portfolio_value = (current_shares * current_day_prices).sum() + remaining_cash
            
            # Calcular nuevas participaciones basadas en pesos óptimos
            new_shares = calculate_shares(optimal_weights, current_day_prices, current_portfolio_value)
            
            # Calcular costes de transacción
            transaction_cost = calculate_transaction_costs(current_shares, new_shares, current_day_prices)
            
            # Actualizar el valor de la cartera después de los costes
            remaining_cash = current_portfolio_value - (new_shares * current_day_prices).sum() - transaction_cost
            
            # Actualizar las participaciones actuales
            current_shares = new_shares
            
            # Calcular los valores diarios de la cartera durante el mes actual (out-of-sample)
            current_month_prices = prices_df.loc[(prices_df.index >= current_month_start) & (prices_df.index < next_month_start)]
            
            for date, prices in current_month_prices.iterrows():
                portfolio_value = (current_shares * prices).sum() + remaining_cash
                portfolio_values[date] = portfolio_value
                
                # Calcular retorno diario (a partir del segundo día)
                if len(portfolio_values) > 1:
                    prev_date = list(portfolio_values.keys())[-2]
                    daily_return = portfolio_value / portfolio_values[prev_date] - 1
                    all_returns.append((date, daily_return))
    
    # Convertir los retornos a DataFrame
    returns_df = pd.DataFrame([r[1] for r in all_returns], index=[r[0] for r in all_returns], columns=['Return'])
    
    # Calcular métricas finales
    final_metrics = calculate_metrics(returns_df['Return'])
    
    # Convertir valores del portafolio a DataFrame
    portfolio_values_df = pd.DataFrame(list(portfolio_values.items()), columns=['Date', 'Value'])
    portfolio_values_df.set_index('Date', inplace=True)
    
    # Convertir historial de pesos a DataFrame
    weights_df = pd.DataFrame(weights_history).T
    weights_df.columns = prices_df.columns
    
    return portfolio_values_df, returns_df, final_metrics, weights_df

# Ejecutar la simulación con peso mínimo del 5%
portfolio_values, returns, metrics, weights_history = run_simulation(df, min_weight=0.05)

# Guardar resultados
portfolio_values.to_csv('results/markowitz_portfolio_values.csv')
returns.to_csv('results/daily_returns.csv')
weights_history.to_csv('results/weights_history.csv')

# Guardar métricas en CSV
metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv('results/portfolio_metrics.csv')

# Visualizar la evolución del valor de la cartera
plt.figure(figsize=(12, 6))
plt.plot(portfolio_values.index, portfolio_values['Value'])
plt.title('Evolución del Valor de la Cartera')
plt.xlabel('Fecha')
plt.ylabel('Valor (USD)')
plt.grid(True)
plt.savefig('results/markowitz_portfolio_evolution.png')

# Visualizar la evolución de los pesos
plt.figure(figsize=(12, 6))
weights_history.plot(figsize=(12, 6))
plt.title('Evolución de los Pesos de la Cartera')
plt.xlabel('Fecha')
plt.ylabel('Peso (%)')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True)
plt.savefig('results/weights_evolution.png')

# Mostrar métricas finales
print("Métricas de la cartera:")
for metric, value in metrics.items():
    if '%' in metric:
        print(f"{metric}: {value:.2f}%")
    else:
        print(f"{metric}: {value:.4f}")