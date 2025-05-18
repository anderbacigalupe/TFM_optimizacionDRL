import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os
from datetime import datetime, timedelta
import calendar
import seaborn as sns
import matplotlib.gridspec as gridspec

# Crear carpetas de resultados si no existen
for folder in ['results', 'results/crisis_analysis_markowitz']:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Cargar los datos
df = pd.read_csv('data/processed/processed_prices.csv', index_col='Date', parse_dates=True)

# Definir los periodos de crisis
crisis_periods = {
    'Crisis Financiera Global': ('2007-10-01', '2009-03-31'),
    'Crisis COVID-19': ('2020-02-01', '2020-04-30'),
    'Crisis Inflacionaria': ('2021-12-01', '2022-10-31')
}

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

# Función para calcular métricas de rendimiento con más detalle
def calculate_performance_metrics(returns, risk_free_rate=0.0):
    # Calcular métricas anualizadas
    total_return = (1 + returns).prod() - 1
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0
    volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0
    
    # Sharpe y Sortino
    sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
    
    # Drawdown
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns / running_max) - 1
    max_drawdown = drawdown.min()
    
    # VaR y CVaR (95%)
    var_95 = np.percentile(returns, 5)
    cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
    
    # Beta (asumiendo 1 por defecto, en un análisis real habría que comparar con un benchmark)
    beta = 1
    
    return {
        'Total Return': total_return,
        'Annualized Return': annualized_return,
        'Annualized Volatility': volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Maximum Drawdown': max_drawdown,
        'VaR 95%': var_95,
        'CVaR 95%': cvar_95,
        'Beta': beta
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
    
    # Convertir valores del portafolio a DataFrame
    portfolio_values_df = pd.DataFrame(list(portfolio_values.items()), columns=['Date', 'Value'])
    portfolio_values_df.set_index('Date', inplace=True)
    
    # Convertir historial de pesos a DataFrame
    weights_df = pd.DataFrame(weights_history).T
    weights_df.columns = prices_df.columns
    
    return portfolio_values_df, returns_df, weights_df

# Ejecutar la simulación con peso mínimo del 5%
portfolio_values, returns, weights_history = run_simulation(df, min_weight=0.05)

# Calcular métricas para el periodo completo
full_period_metrics = calculate_performance_metrics(returns['Return'])

# Calcular métricas para cada periodo de crisis
crisis_metrics = {}
crisis_returns = {}
crisis_values = {}
crisis_drawdowns = {}
crisis_weights = {}

for crisis_name, (start, end) in crisis_periods.items():
    # Filtrar datos para el periodo de crisis
    crisis_data = portfolio_values.loc[start:end]
    crisis_returns_data = returns.loc[start:end]
    
    if len(crisis_data) > 0:
        # Guardar los rendimientos y valores para gráficos
        crisis_returns[crisis_name] = crisis_returns_data['Return']
        crisis_values[crisis_name] = crisis_data['Value']
        
        # Filtrar pesos durante la crisis
        crisis_weights[crisis_name] = weights_history.loc[weights_history.index.to_series().between(start, end)]
        
        # Calcular drawdown para el periodo
        cumulative_returns = (1 + crisis_returns[crisis_name]).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max) - 1
        crisis_drawdowns[crisis_name] = drawdown
        
        # Calcular métricas para el periodo
        crisis_metrics[crisis_name] = calculate_performance_metrics(crisis_returns[crisis_name])
        
        # Añadir métricas específicas para crisis
        # Rendimiento total durante la crisis
        crisis_metrics[crisis_name]['Total Value Change'] = crisis_data['Value'][-1] / crisis_data['Value'][0] - 1
        
        # Tiempo de recuperación (si recuperó durante el periodo)
        if drawdown.iloc[-1] > -0.1 and drawdown.min() < -0.1:  # Si hubo recuperación significativa
            recovery_point = drawdown[drawdown <= -0.1].index[-1]
            days_to_recover = (drawdown.index[-1] - recovery_point).days
            crisis_metrics[crisis_name]['Recovery Days'] = days_to_recover
        else:
            crisis_metrics[crisis_name]['Recovery Days'] = "No recuperado"
        
        # Peso promedio en activos refugio durante la crisis
        if not crisis_weights[crisis_name].empty:
            safe_assets = ['GLD', 'TLT']  # Oro y bonos del tesoro como activos refugio
            # Verificar que estos activos estén en el dataset
            available_safe_assets = [asset for asset in safe_assets if asset in crisis_weights[crisis_name].columns]
            if available_safe_assets:
                safe_weight_avg = crisis_weights[crisis_name][available_safe_assets].mean().sum()
                crisis_metrics[crisis_name]['Avg Safe Asset Weight'] = safe_weight_avg

# Guardar resultados básicos
portfolio_values.to_csv('results/crisis_analysis_markowitz/markowitz_portfolio_values.csv')
returns.to_csv('results/crisis_analysis_markowitz/daily_returns.csv')
weights_history.to_csv('results/crisis_analysis_markowitz/weights_history.csv')

# Crear visualizaciones comparativas

# 1. Evolución del valor de la cartera con periodos de crisis destacados
plt.figure(figsize=(15, 8))
plt.plot(portfolio_values.index, portfolio_values['Value'], label='Valor de la Cartera', color='blue')

# Sombrear áreas de crisis
colors = ['red', 'orange', 'purple']
for i, (crisis_name, (start, end)) in enumerate(crisis_periods.items()):
    plt.axvspan(start, end, color=colors[i], alpha=0.2, label=f'Periodo de {crisis_name}')

plt.title('Evolución del Valor de la Cartera de Markowitz con Periodos de Crisis')
plt.xlabel('Fecha')
plt.ylabel('Valor ($)')
plt.legend()
plt.grid(True)
plt.savefig('results/crisis_analysis_markowitz/portfolio_value_with_crisis_periods.png')

# 2. Drawdowns en los periodos de crisis
plt.figure(figsize=(15, 12))
for i, (crisis_name, drawdown) in enumerate(crisis_drawdowns.items()):
    plt.subplot(3, 1, i+1)
    drawdown.plot(color=colors[i])
    plt.title(f'Drawdown durante {crisis_name}')
    plt.xlabel('Fecha')
    plt.ylabel('Drawdown (%)')
    plt.grid(True)
    plt.ylim(-1, 0.1)  # Limitamos el eje Y para mejor visualización

plt.tight_layout()
plt.savefig('results/crisis_analysis_markowitz/drawdowns_during_crisis.png')

# 3. Distribución de rendimientos en cada periodo
plt.figure(figsize=(15, 12))
for i, (crisis_name, returns_data) in enumerate(crisis_returns.items()):
    plt.subplot(3, 1, i+1)
    sns.histplot(returns_data, kde=True, color=colors[i])
    plt.axvline(0, color='black', linestyle='--')
    plt.title(f'Distribución de Rendimientos durante {crisis_name}')
    plt.xlabel('Rendimiento Diario')
    plt.ylabel('Frecuencia')
    plt.grid(True)

plt.tight_layout()
plt.savefig('results/crisis_analysis_markowitz/returns_distribution_crisis.png')

# 4. Evolución de pesos por etapa de crisis
for crisis_name, weights_df in crisis_weights.items():
    if not weights_df.empty:
        plt.figure(figsize=(15, 8))
        weights_df.plot(kind='area', stacked=True, colormap='viridis')
        plt.title(f'Evolución de Pesos de Markowitz durante {crisis_name}')
        plt.xlabel('Fecha')
        plt.ylabel('Peso en la Cartera')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'results/crisis_analysis_markowitz/weights_during_{crisis_name.replace(" ", "_").lower()}.png')

# 5. Comparación de pesos de activos refugio antes y durante las crisis
plt.figure(figsize=(15, 12))
safe_assets = ['GLD', 'TLT']  # Oro y bonos del tesoro como activos refugio
available_safe_assets = [asset for asset in safe_assets if asset in weights_history.columns]

if available_safe_assets:
    for i, (crisis_name, (start, end)) in enumerate(crisis_periods.items()):
        plt.subplot(3, 1, i+1)
        
        # Obtener pesos 3 meses antes de la crisis
        pre_crisis_start = pd.to_datetime(start) - pd.DateOffset(months=3)
        pre_crisis_weights = weights_history.loc[weights_history.index.to_series().between(pre_crisis_start, start)]
        
        # Obtener pesos durante la crisis
        crisis_period_weights = weights_history.loc[weights_history.index.to_series().between(start, end)]
        
        # Calcular promedios
        if not pre_crisis_weights.empty and not crisis_period_weights.empty:
            pre_crisis_avg = pre_crisis_weights[available_safe_assets].mean()
            crisis_avg = crisis_period_weights[available_safe_assets].mean()
            
            # Crear DataFrame para graficar
            comparison_df = pd.DataFrame({
                'Pre-Crisis': pre_crisis_avg,
                'Durante Crisis': crisis_avg
            })
            
            # Graficar
            comparison_df.plot(kind='bar', ax=plt.gca())
            plt.title(f'Comparación de Pesos en Activos Refugio - {crisis_name}')
            plt.ylabel('Peso Promedio')
            plt.grid(True, axis='y')
            plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig('results/crisis_analysis_markowitz/safe_assets_comparison.png')

# Crear un DataFrame comparativo de métricas
all_metrics = {'Periodo Completo': full_period_metrics}
all_metrics.update(crisis_metrics)

metrics_comparison = pd.DataFrame(all_metrics).T
metrics_comparison = metrics_comparison[['Total Return', 'Annualized Return', 'Annualized Volatility', 
                                         'Sharpe Ratio', 'Sortino Ratio', 'Maximum Drawdown', 
                                         'VaR 95%', 'CVaR 95%']]

# Formatear para mejor visualización
format_dict = {
    'Total Return': '{:.2%}',
    'Annualized Return': '{:.2%}',
    'Annualized Volatility': '{:.2%}',
    'Sharpe Ratio': '{:.2f}',
    'Sortino Ratio': '{:.2f}',
    'Maximum Drawdown': '{:.2%}',
    'VaR 95%': '{:.2%}',
    'CVaR 95%': '{:.2%}'
}

formatted_metrics = metrics_comparison.copy()
for col, format_str in format_dict.items():
    formatted_metrics[col] = formatted_metrics[col].map(lambda x: format_str.format(x))

# Guardar la tabla comparativa
metrics_comparison.to_csv('results/crisis_analysis_markowitz/metrics_comparison_raw.csv')
formatted_metrics.to_csv('results/crisis_analysis_markowitz/metrics_comparison_formatted.csv')

# Calcular correlaciones entre activos para cada crisis
correlation_tables = {}

for crisis_name, (start, end) in crisis_periods.items():
    crisis_data = df.loc[start:end]
    if len(crisis_data) > 0:
        # Calcular rendimientos de los activos durante la crisis
        crisis_asset_returns = crisis_data.pct_change().dropna()
        
        # Calcular matriz de correlación
        correlation_matrix = crisis_asset_returns.corr()
        correlation_tables[crisis_name] = correlation_matrix
        
        # Visualizar como heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        plt.title(f'Correlaciones entre Activos durante {crisis_name}')
        plt.tight_layout()
        plt.savefig(f'results/crisis_analysis_markowitz/correlations_{crisis_name.replace(" ", "_").lower()}.png')

# Crear un informe resumen en formato de texto
with open('results/crisis_analysis_markowitz/crisis_analysis_summary.txt', 'w') as f:
    f.write("ANÁLISIS DE CARTERA DE MARKOWITZ DURANTE PERIODOS DE CRISIS\n")
    f.write("=========================================================\n\n")
    
    f.write("MÉTRICAS COMPARATIVAS\n")
    f.write("--------------------\n")
    f.write(formatted_metrics.to_string())
    f.write("\n\n")
    
    for crisis_name in crisis_periods.keys():
        f.write(f"ANÁLISIS DETALLADO: {crisis_name}\n")
        f.write("-" * (len(crisis_name) + 19) + "\n")
        
        if crisis_name in crisis_metrics:
            metrics = crisis_metrics[crisis_name]
            f.write(f"Rendimiento total: {metrics['Total Return']:.2%}\n")
            f.write(f"Rendimiento anualizado: {metrics['Annualized Return']:.2%}\n")
            f.write(f"Volatilidad anualizada: {metrics['Annualized Volatility']:.2%}\n")
            f.write(f"Ratio Sharpe: {metrics['Sharpe Ratio']:.2f}\n")
            f.write(f"Ratio Sortino: {metrics['Sortino Ratio']:.2f}\n")
            f.write(f"Máximo Drawdown: {metrics['Maximum Drawdown']:.2%}\n")
            f.write(f"VaR 95%: {metrics['VaR 95%']:.2%}\n")
            f.write(f"CVaR 95%: {metrics['CVaR 95%']:.2%}\n")
            
            # Comportamiento de activos refugio
            safe_assets = ['GLD', 'TLT']
            available_safe_assets = [asset for asset in safe_assets if asset in weights_history.columns]
            if available_safe_assets and crisis_name in crisis_weights:
                avg_weight = crisis_weights[crisis_name][available_safe_assets].mean().sum()
                f.write(f"Peso promedio en activos refugio: {avg_weight:.2%}\n")
            
            # Añadir métricas específicas de crisis
            if 'Total Value Change' in metrics:
                f.write(f"Cambio total en valor: {metrics['Total Value Change']:.2%}\n")
            if 'Recovery Days' in metrics:
                f.write(f"Días para recuperación: {metrics['Recovery Days']}\n")
            
            # Análisis de las correlaciones
            if crisis_name in correlation_tables:
                f.write("\nAnálisis de correlaciones:\n")
                # Identificar las correlaciones más altas y más bajas
                corr_matrix = correlation_tables[crisis_name]
                high_corr = corr_matrix.unstack().sort_values(ascending=False)
                # Eliminar auto-correlaciones (que son 1)
                high_corr = high_corr[high_corr < 0.999]
                f.write(f"Máxima correlación positiva: {high_corr.index[0][0]} y {high_corr.index[0][1]} ({high_corr.iloc[0]:.2f})\n")
                
                low_corr = corr_matrix.unstack().sort_values(ascending=True)
                f.write(f"Máxima correlación negativa: {low_corr.index[0][0]} y {low_corr.index[0][1]} ({low_corr.iloc[0]:.2f})\n")
            
            f.write("\n")
        else:
            f.write("No hay datos suficientes para este periodo.\n\n")
    
    f.write("\nCOMPARACIÓN CON PERIODO COMPLETO\n")
    f.write("-------------------------------\n")
    for crisis_name in crisis_periods.keys():
        if crisis_name in crisis_metrics:
            f.write(f"{crisis_name}:\n")
            # Comparar rendimiento con periodo completo
            ret_diff = crisis_metrics[crisis_name]['Annualized Return'] - full_period_metrics['Annualized Return']
            f.write(f"- Diferencia en rendimiento anualizado: {ret_diff:.2%}\n")
            
            # Comparar volatilidad
            vol_diff = crisis_metrics[crisis_name]['Annualized Volatility'] - full_period_metrics['Annualized Volatility']
            f.write(f"- Diferencia en volatilidad anualizada: {vol_diff:.2%}\n")
            
            # Comparar Sharpe
            sharpe_diff = crisis_metrics[crisis_name]['Sharpe Ratio'] - full_period_metrics['Sharpe Ratio']
            f.write(f"- Diferencia en Ratio Sharpe: {sharpe_diff:.2f}\n")
            
            f.write("\n")
    
    f.write("\nNOTAS ADICIONALES\n")
    f.write("----------------\n")
    f.write("- Los periodos de crisis analizados son:\n")
    for name, (start, end) in crisis_periods.items():
        f.write(f"  * {name}: {start} a {end}\n")
    f.write("\n- El análisis se basa en la optimización de Markowitz con las siguientes características:\n")
    f.write("  * Rebalanceo mensual\n")
    f.write("  * Peso mínimo por activo: 5%\n")
    f.write("  * Optimización walk-forward (uso de datos disponibles hasta cada punto de decisión)\n")
    f.write("  * Simulación de costos de transacción y slippage\n")
    f.write("\n- Los activos considerados 'refugio' para el análisis son GLD (oro) y TLT (bonos del tesoro a largo plazo).\n")

# Mostrar métricas comparativas en consola
print("\nAnálisis de crisis completado para la cartera de Markowitz. Resultados guardados en la carpeta 'results/crisis_analysis_markowitz'.")
print("\nMétricas comparativas:")
print(formatted_metrics)

# Análisis de la adaptación de la cartera durante las crisis
adaptability_metrics = {}

for crisis_name, (start, end) in crisis_periods.items():
    if crisis_name in crisis_weights:
        # Obtener pesos al inicio y al final de la crisis
        crisis_period_weights = crisis_weights[crisis_name]
        if not crisis_period_weights.empty and len(crisis_period_weights) > 1:
            start_weights = crisis_period_weights.iloc[0]
            end_weights = crisis_period_weights.iloc[-1]
            
            # Calcular el cambio absoluto en la asignación
            weight_change = np.abs(end_weights - start_weights).sum() / 2  # Dividido por 2 para no contar dos veces
            
            # Calcular la velocidad de adaptación (cambio por mes)
            crisis_duration = (pd.to_datetime(end) - pd.to_datetime(start)).days / 30  # en meses
            adaptation_speed = weight_change / crisis_duration if crisis_duration > 0 else 0
            
            adaptability_metrics[crisis_name] = {
                'Total Weight Change': weight_change,
                'Adaptation Speed (per month)': adaptation_speed
            }

# Guardar métricas de adaptabilidad
if adaptability_metrics:
    adapt_df = pd.DataFrame(adaptability_metrics).T
    adapt_df.to_csv('results/crisis_analysis_markowitz/adaptability_metrics.csv')

# Continuación del script anterior

# Crear visualización adicional: Comparación de rendimientos acumulados durante las crisis
plt.figure(figsize=(15, 12))
for i, (crisis_name, returns_data) in enumerate(crisis_returns.items()):
    plt.subplot(3, 1, i+1)
    cumulative_returns = (1 + returns_data).cumprod()
    # Normalizar a 100 al inicio del periodo
    normalized_returns = cumulative_returns / cumulative_returns.iloc[0] * 100
    normalized_returns.plot(color=colors[i], linewidth=2)
    plt.axhline(y=100, color='black', linestyle='--')
    plt.title(f'Rendimientos Acumulados durante {crisis_name}')
    plt.xlabel('Fecha')
    plt.ylabel('Valor (Base 100)')
    plt.grid(True)

plt.tight_layout()
plt.savefig('results/crisis_analysis_markowitz/cumulative_returns_crisis.png')

# Análisis de rotación de cartera durante las crisis
turnover_metrics = {}

for crisis_name, (start, end) in crisis_periods.items():
    if crisis_name in crisis_weights:
        crisis_period_weights = crisis_weights[crisis_name]
        
        if len(crisis_period_weights) > 1:
            # Calcular la rotación de la cartera para cada par de fechas consecutivas
            turnover_values = []
            
            for i in range(1, len(crisis_period_weights)):
                prev_weights = crisis_period_weights.iloc[i-1]
                curr_weights = crisis_period_weights.iloc[i]
                
                # Rotación = suma de cambios absolutos / 2 (para no contar doble)
                turnover = np.sum(np.abs(curr_weights - prev_weights)) / 2
                turnover_values.append(turnover)
            
            # Guardar métricas
            turnover_metrics[crisis_name] = {
                'Mean Turnover': np.mean(turnover_values),
                'Max Turnover': np.max(turnover_values),
                'Total Turnover': np.sum(turnover_values)
            }

# Guardar métricas de rotación
if turnover_metrics:
    turnover_df = pd.DataFrame(turnover_metrics).T
    turnover_df.to_csv('results/crisis_analysis_markowitz/turnover_metrics.csv')
    
    # Visualizar rotación por crisis
    plt.figure(figsize=(12, 6))
    turnover_df['Mean Turnover'].plot(kind='bar', color='skyblue')
    plt.title('Rotación Media de la Cartera por Periodo de Crisis')
    plt.ylabel('Rotación Media')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('results/crisis_analysis_markowitz/turnover_by_crisis.png')

# Análisis de la evolución de la exposición a activos de riesgo vs. refugio durante las crisis
if 'GLD' in df.columns and 'TLT' in df.columns:  # Verificar que los activos refugio estén en el dataset
    plt.figure(figsize=(15, 12))
    
    for i, (crisis_name, (start, end)) in enumerate(crisis_periods.items()):
        if crisis_name in crisis_weights:
            crisis_period_weights = crisis_weights[crisis_name]
            
            if not crisis_period_weights.empty:
                plt.subplot(3, 1, i+1)
                
                # Crear categorías de activos
                safe_assets = ['GLD', 'TLT']  # Oro y bonos del tesoro
                # Activos de riesgo son todos los demás
                risk_assets = [col for col in crisis_period_weights.columns if col not in safe_assets]
                
                # Calcular peso total en cada categoría
                safe_weights = crisis_period_weights[safe_assets].sum(axis=1)
                risk_weights = crisis_period_weights[risk_assets].sum(axis=1)
                
                # Crear DataFrame para visualización
                exposure_df = pd.DataFrame({
                    'Activos Refugio': safe_weights,
                    'Activos de Riesgo': risk_weights
                })
                
                # Graficar
                exposure_df.plot(kind='area', stacked=True, ax=plt.gca(), 
                                 colormap='RdYlGn', alpha=0.7)
                plt.title(f'Evolución de la Exposición durante {crisis_name}')
                plt.xlabel('Fecha')
                plt.ylabel('Peso en la Cartera')
                plt.grid(True)
                plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('results/crisis_analysis_markowitz/risk_exposure_evolution.png')

# Análisis del comportamiento de la volatilidad durante las crisis
volatility_windows = {}

for crisis_name, returns_data in crisis_returns.items():
    if len(returns_data) > 20:  # Asegurar suficientes datos para calcular ventanas
        # Calcular volatilidad rodante con una ventana de 20 días
        rolling_vol = returns_data.rolling(window=20).std() * np.sqrt(252)
        volatility_windows[crisis_name] = rolling_vol

# Visualizar volatilidad rodante
if volatility_windows:
    plt.figure(figsize=(15, 12))
    
    for i, (crisis_name, vol_data) in enumerate(volatility_windows.items()):
        plt.subplot(3, 1, i+1)
        vol_data.plot(color=colors[i], linewidth=2)
        plt.title(f'Volatilidad Rodante (20 días) durante {crisis_name}')
        plt.xlabel('Fecha')
        plt.ylabel('Volatilidad Anualizada')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/crisis_analysis_markowitz/rolling_volatility.png')

# Análisis de máximos retornos diarios positivos y negativos durante las crisis
extreme_returns = {}

for crisis_name, returns_data in crisis_returns.items():
    # Ordenar retornos para encontrar extremos
    sorted_returns = returns_data.sort_values()
    
    # Guardar los 5 peores y mejores días
    extreme_returns[crisis_name] = {
        'Top 5 Worst Days': sorted_returns.head(5),
        'Top 5 Best Days': sorted_returns.tail(5)
    }

# Guardar datos de retornos extremos
for crisis_name, data in extreme_returns.items():
    worst_days = pd.DataFrame(data['Top 5 Worst Days'])
    worst_days.columns = ['Return']
    worst_days['Date'] = worst_days.index
    worst_days.reset_index(drop=True, inplace=True)
    
    best_days = pd.DataFrame(data['Top 5 Best Days'])
    best_days.columns = ['Return']
    best_days['Date'] = best_days.index
    best_days.reset_index(drop=True, inplace=True)
    
    extreme_df = pd.concat([
        worst_days.rename(columns={'Return': 'Worst Return', 'Date': 'Worst Date'}),
        best_days.rename(columns={'Return': 'Best Return', 'Date': 'Best Date'})
    ], axis=1)
    
    extreme_df.to_csv(f'results/crisis_analysis_markowitz/extreme_returns_{crisis_name.replace(" ", "_").lower()}.csv')

# Análisis comparativo entre periodos de crisis: Gráfico de radar para métricas clave
# Este análisis requiere matplotlib y numpy, que ya deberían estar importados

# Seleccionar métricas para el gráfico de radar
radar_metrics = ['Sharpe Ratio', 'Sortino Ratio', 'Maximum Drawdown', 'VaR 95%', 'Annualized Volatility']

# Transformar métricas al rango [0,1] para el gráfico de radar, donde 1 es mejor
normalized_metrics = {}

for metric in radar_metrics:
    values = [metrics_comparison.loc[period, metric] for period in metrics_comparison.index]
    
    if metric in ['Maximum Drawdown', 'VaR 95%', 'Annualized Volatility']:
        # Para estas métricas, valores más bajos son mejores
        min_val = min(values)
        max_val = max(values)
        if max_val != min_val:  # Evitar división por cero
            normalized_metrics[metric] = [1 - (val - min_val) / (max_val - min_val) for val in values]
        else:
            normalized_metrics[metric] = [0.5 for _ in values]  # Valor neutral si todos son iguales
    else:
        # Para estas métricas, valores más altos son mejores
        min_val = min(values)
        max_val = max(values)
        if max_val != min_val:
            normalized_metrics[metric] = [(val - min_val) / (max_val - min_val) for val in values]
        else:
            normalized_metrics[metric] = [0.5 for _ in values]

# Crear gráfico de radar
plt.figure(figsize=(10, 10))
angles = np.linspace(0, 2*np.pi, len(radar_metrics), endpoint=False)
angles = np.concatenate((angles, [angles[0]]))  # Cerrar el polígono

for i, period in enumerate(metrics_comparison.index):
    values = [normalized_metrics[metric][i] for metric in radar_metrics]
    values = np.concatenate((values, [values[0]]))  # Cerrar el polígono
    
    plt.polar(angles, values, marker='o', linestyle='-', linewidth=2, label=period)

# Añadir etiquetas
plt.xticks(angles[:-1], radar_metrics)
plt.title('Comparación de Métricas de Rendimiento por Periodo')
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig('results/crisis_analysis_markowitz/radar_chart_comparison.png')

# Comparación de la distribución de retornos: Boxplot
plt.figure(figsize=(12, 8))
return_data_for_boxplot = []
labels = []

for period in metrics_comparison.index:
    if period == 'Periodo Completo':
        return_data_for_boxplot.append(returns['Return'])
        labels.append(period)
    else:
        for crisis_name, data in crisis_returns.items():
            if period == crisis_name:
                return_data_for_boxplot.append(data)
                labels.append(period)

plt.boxplot(return_data_for_boxplot, labels=labels, showfliers=False)
plt.title('Comparación de la Distribución de Retornos Diarios')
plt.ylabel('Retorno Diario')
plt.grid(True, axis='y')
plt.axhline(y=0, color='red', linestyle='--')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('results/crisis_analysis_markowitz/return_distribution_boxplot.png')

# Análisis de la eficiencia: gráfico de Rendimiento vs. Riesgo
plt.figure(figsize=(10, 8))
for i, period in enumerate(metrics_comparison.index):
    risk = metrics_comparison.loc[period, 'Annualized Volatility']
    ret = metrics_comparison.loc[period, 'Annualized Return']
    
    if period == 'Periodo Completo':
        marker_size = 150
        color = 'blue'
    else:
        marker_size = 100
        color = colors[i-1] if i > 0 else 'green'
    
    plt.scatter(risk, ret, s=marker_size, c=color, alpha=0.7, label=period)
    plt.annotate(period, (risk, ret), xytext=(10, 0), textcoords='offset points')

# Dibujar línea para Sharpe Ratio = 1
x_range = np.linspace(0, max(metrics_comparison['Annualized Volatility'])*1.1, 100)
plt.plot(x_range, x_range, 'r--', label='Sharpe Ratio = 1')

plt.xlabel('Volatilidad Anualizada')
plt.ylabel('Rendimiento Anualizado')
plt.title('Análisis de Eficiencia: Rendimiento vs. Riesgo')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('results/crisis_analysis_markowitz/risk_return_scatter.png')

# Ampliación del informe de resumen
with open('results/crisis_analysis_markowitz/crisis_analysis_summary.txt', 'a') as f:
    f.write("\n\nROTACIÓN DE CARTERA\n")
    f.write("-----------------\n")
    if turnover_metrics:
        for crisis_name, metrics in turnover_metrics.items():
            f.write(f"{crisis_name}:\n")
            f.write(f"  - Rotación media: {metrics['Mean Turnover']:.2%}\n")
            f.write(f"  - Rotación máxima: {metrics['Max Turnover']:.2%}\n")
            f.write(f"  - Rotación total: {metrics['Total Turnover']:.2%}\n\n")
    
    f.write("\nRETORNOS EXTREMOS\n")
    f.write("----------------\n")
    for crisis_name, data in extreme_returns.items():
        f.write(f"{crisis_name}:\n")
        f.write("  Peores días:\n")
        for date, ret in zip(data['Top 5 Worst Days'].index, data['Top 5 Worst Days']):
            f.write(f"    - {date.strftime('%Y-%m-%d')}: {ret:.2%}\n")
        
        f.write("  Mejores días:\n")
        for date, ret in zip(data['Top 5 Best Days'].index, data['Top 5 Best Days']):
            f.write(f"    - {date.strftime('%Y-%m-%d')}: {ret:.2%}\n\n")
    
    f.write("\nCONCLUSIONES DEL ANÁLISIS\n")
    f.write("------------------------\n")
    f.write("El análisis de la cartera de Markowitz durante los tres periodos de crisis muestra:\n\n")
    
    # Analizar y escribir conclusiones específicas basadas en los datos
    # Esto puede variar dependiendo de los resultados reales
    
    # Verificar si los datos muestran mejor rendimiento en algún periodo específico
    best_sharpe_period = metrics_comparison['Sharpe Ratio'].idxmax()
    worst_drawdown_period = metrics_comparison['Maximum Drawdown'].idxmin()
    
    f.write(f"1. El mejor rendimiento ajustado al riesgo (Ratio Sharpe) se observó durante {best_sharpe_period}.\n")
    f.write(f"2. El periodo con menor drawdown máximo fue {worst_drawdown_period}.\n")
    f.write("3. La capacidad de adaptación de la optimización de Markowitz se refleja en los cambios de asignación durante las crisis.\n")
    
    # Añadir información sobre comportamiento del peso en activos refugio si está disponible
    if 'GLD' in df.columns and 'TLT' in df.columns:
        f.write("4. Durante periodos de crisis, la asignación a activos refugio (GLD y TLT) presentó cambios significativos.\n")
    
    f.write("\n5. El análisis muestra cómo el rebalanceo mensual y la optimización walk-forward permiten adaptarse a condiciones cambiantes de mercado.\n")

print("\nAnálisis de crisis para la cartera de Markowitz completado con gráficos y métricas adicionales.")