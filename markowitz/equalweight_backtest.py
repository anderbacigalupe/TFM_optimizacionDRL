import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import seaborn as sns

# Crear carpetas de resultados si no existen
for folder in ['results', 'results/crisis_analysis']:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Cargar los datos
file_path = os.path.join('data', 'processed', 'processed_prices.csv')
etf_data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')

# Filtrar por fechas
start_date = '2007-04-11'
end_date = '2025-04-10'
etf_data = etf_data.loc[start_date:end_date]

# Verificar los ETFs disponibles
print(f"ETFs disponibles: {etf_data.columns.tolist()}")
print(f"Período de análisis: {etf_data.index[0]} a {etf_data.index[-1]}")

# Tomar los primeros 6 ETFs para la cartera equiponderada
if len(etf_data.columns) >= 6:
    selected_etfs = etf_data.columns[:6]
else:
    selected_etfs = etf_data.columns
    print(f"Advertencia: Solo hay {len(selected_etfs)} ETFs disponibles")

print(f"ETFs seleccionados: {selected_etfs}")

# Definir los periodos de crisis
crisis_periods = {
    'Crisis Financiera Global': ('2007-10-01', '2009-03-31'),
    'Crisis COVID-19': ('2020-02-01', '2020-04-30'),
    'Crisis Inflacionaria': ('2021-12-01', '2022-10-31')
}

# Parámetros iniciales
initial_capital = 1_000_000
trading_days_per_year = 252

# Función para calcular el número de participaciones a comprar en rebalanceo equiponderado
def calculate_shares_equal_weight(total_capital, prices):
    n_etfs = len(prices)
    capital_per_etf = total_capital / n_etfs
    shares = np.floor(capital_per_etf / prices).astype(int)
    return shares

# Función para calcular los costos de transacción
def calculate_transaction_costs(shares_to_trade, prices):
    # Costos por participación
    cost_per_share = 0.0035
    min_cost_per_operation = 0.35
    max_cost_percentage = 0.01
    slippage_percentage = 0.001
    
    # Aplicar slippage al precio
    adjusted_prices_buy = prices * (1 + slippage_percentage)
    adjusted_prices_sell = prices * (1 - slippage_percentage)
    
    # Separar compras y ventas
    buy_shares = np.maximum(shares_to_trade, 0)
    sell_shares = np.maximum(-shares_to_trade, 0)
    
    # Costo total por slippage
    slippage_costs = sum(buy_shares * prices * slippage_percentage) + sum(sell_shares * prices * slippage_percentage)
    
    # Costos de comisión
    buy_commission = 0
    for i, shares in enumerate(buy_shares):
        if shares > 0:
            commission = shares * cost_per_share
            max_commission = prices[i] * shares * max_cost_percentage
            buy_commission += min(max(commission, min_cost_per_operation), max_commission)
    
    sell_commission = 0
    for i, shares in enumerate(sell_shares):
        if shares > 0:
            commission = shares * cost_per_share
            max_commission = prices[i] * shares * max_cost_percentage
            sell_commission += min(max(commission, min_cost_per_operation), max_commission)
    
    total_commission = buy_commission + sell_commission
    
    return total_commission + slippage_costs, adjusted_prices_buy, adjusted_prices_sell

# Función para calcular métricas de rendimiento
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
    
    # Beta (usando S&P 500 como aproximación)
    # Para un análisis más preciso, deberías cargar datos del S&P 500
    # benchmark_returns = ...
    # covariance = np.cov(returns, benchmark_returns)[0, 1]
    # benchmark_variance = np.var(benchmark_returns)
    # beta = covariance / benchmark_variance if benchmark_variance > 0 else 1
    beta = 1  # Valor por defecto para este ejemplo
    
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

# Inicialización del seguimiento de cartera
portfolio_values = []
current_shares = None
available_cash = initial_capital
transaction_costs_history = []
portfolio_weights_history = []
rebalance_dates = []

# Obtener el primer día de cada mes en el período de datos
monthly_dates = etf_data.index.to_period('M').drop_duplicates().to_timestamp()

# Simulación de inversión y seguimiento diario
for date, row in etf_data.iterrows():
    prices = row[selected_etfs].values
    
    # Compra inicial o rebalanceo mensual
    if current_shares is None or date in monthly_dates.values:
        if current_shares is not None:
            # Es un rebalanceo, guardamos la fecha
            rebalance_dates.append(date)
        
        # Valor actual de la cartera antes del rebalanceo
        current_portfolio_value = 0 if current_shares is None else sum(current_shares * prices) + available_cash
        
        # Calcular nuevas participaciones para rebalanceo equiponderado
        new_shares = calculate_shares_equal_weight(
            initial_capital if current_shares is None else current_portfolio_value, 
            prices
        )
        
        # Calcular cambios en participaciones
        shares_to_trade = new_shares if current_shares is None else new_shares - current_shares
        
        # Calcular costos de transacción
        transaction_costs, adj_prices_buy, adj_prices_sell = calculate_transaction_costs(shares_to_trade, prices)
        transaction_costs_history.append(transaction_costs)
        
        # Calcular flujo de efectivo por compras/ventas
        cash_flow = 0
        for i, trade in enumerate(shares_to_trade):
            if trade > 0:  # Compra
                cash_flow -= trade * adj_prices_buy[i]
            elif trade < 0:  # Venta
                cash_flow += -trade * adj_prices_sell[i]
        
        # Actualizar efectivo disponible
        available_cash = available_cash + cash_flow - transaction_costs
        
        # Actualizar participaciones
        current_shares = new_shares
        
        # Calcular y guardar los pesos del portafolio
        total_value = sum(current_shares * prices) + available_cash
        weights = (current_shares * prices) / total_value if total_value > 0 else np.zeros_like(current_shares)
        portfolio_weights_history.append((date, weights))
    
    # Valor actual de la cartera
    portfolio_value = sum(current_shares * prices) + available_cash
    portfolio_values.append((date, portfolio_value))

# Crear un DataFrame con los valores del portafolio
portfolio_df = pd.DataFrame(portfolio_values, columns=['Date', 'Portfolio Value']).set_index('Date')

# Calcular rendimientos diarios
portfolio_df['Daily Return'] = portfolio_df['Portfolio Value'].pct_change()

# Crear DataFrame con los pesos históricos
weights_df = pd.DataFrame(
    [weights for _, weights in portfolio_weights_history],
    index=[date for date, _ in portfolio_weights_history],
    columns=selected_etfs
)

# Eliminar el primer valor (NaN) para los rendimientos
portfolio_returns = portfolio_df['Daily Return'].dropna()

# Calcular métricas para el periodo completo
full_period_metrics = calculate_performance_metrics(portfolio_returns)

# Calcular métricas para cada periodo de crisis
crisis_metrics = {}
crisis_returns = {}
crisis_values = {}
crisis_drawdowns = {}

for crisis_name, (start, end) in crisis_periods.items():
    # Filtrar datos para el periodo de crisis
    crisis_data = portfolio_df.loc[start:end]
    
    if len(crisis_data) > 0:
        # Calcular rendimientos para el periodo
        crisis_period_returns = crisis_data['Daily Return'].dropna()
        
        # Guardar los rendimientos y valores para gráficos
        crisis_returns[crisis_name] = crisis_period_returns
        crisis_values[crisis_name] = crisis_data['Portfolio Value']
        
        # Calcular drawdown para el periodo
        cumulative_returns = (1 + crisis_period_returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max) - 1
        crisis_drawdowns[crisis_name] = drawdown
        
        # Calcular métricas para el periodo
        crisis_metrics[crisis_name] = calculate_performance_metrics(crisis_period_returns)
        
        # Añadir métricas específicas para crisis
        # Rendimiento total durante la crisis
        crisis_metrics[crisis_name]['Total Value Change'] = crisis_data['Portfolio Value'][-1] / crisis_data['Portfolio Value'][0] - 1
        
        # Tiempo de recuperación (si recuperó durante el periodo)
        if drawdown.iloc[-1] > -0.1 and drawdown.min() < -0.1:  # Si hubo recuperación significativa
            recovery_point = drawdown[drawdown <= -0.1].index[-1]
            days_to_recover = (drawdown.index[-1] - recovery_point).days
            crisis_metrics[crisis_name]['Recovery Days'] = days_to_recover
        else:
            crisis_metrics[crisis_name]['Recovery Days'] = "No recuperado"
        
        # Peso promedio en activos refugio durante la crisis
        if 'GLD' in selected_etfs and 'TLT' in selected_etfs:
            crisis_weights = weights_df.loc[weights_df.index.to_series().between(start, end)]
            if not crisis_weights.empty:
                safe_assets = ['GLD', 'TLT']  # Oro y bonos del tesoro como activos refugio
                safe_weight_avg = crisis_weights[safe_assets].mean().sum()
                crisis_metrics[crisis_name]['Avg Safe Asset Weight'] = safe_weight_avg

# Crear visualizaciones comparativas

# 1. Evolución del valor de la cartera con periodos de crisis destacados
plt.figure(figsize=(15, 8))
plt.plot(portfolio_df.index, portfolio_df['Portfolio Value'], label='Valor de la Cartera', color='blue')

# Sombrear áreas de crisis
colors = ['red', 'orange', 'purple']
for i, (crisis_name, (start, end)) in enumerate(crisis_periods.items()):
    plt.axvspan(start, end, color=colors[i], alpha=0.2, label=f'Periodo de {crisis_name}')

plt.title('Evolución del Valor de la Cartera con Periodos de Crisis')
plt.xlabel('Fecha')
plt.ylabel('Valor ($)')
plt.legend()
plt.grid(True)
plt.savefig('results/crisis_analysis/portfolio_value_with_crisis_periods.png')

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
plt.savefig('results/crisis_analysis/drawdowns_during_crisis.png')

# 3. Distribución de rendimientos en cada periodo
plt.figure(figsize=(15, 12))
for i, (crisis_name, returns) in enumerate(crisis_returns.items()):
    plt.subplot(3, 1, i+1)
    sns.histplot(returns, kde=True, color=colors[i])
    plt.axvline(0, color='black', linestyle='--')
    plt.title(f'Distribución de Rendimientos durante {crisis_name}')
    plt.xlabel('Rendimiento Diario')
    plt.ylabel('Frecuencia')
    plt.grid(True)

plt.tight_layout()
plt.savefig('results/crisis_analysis/returns_distribution_crisis.png')

# 4. Evolución de pesos por etapa de crisis
for crisis_name, (start, end) in crisis_periods.items():
    crisis_weights = weights_df.loc[weights_df.index.to_series().between(start, end)]
    if not crisis_weights.empty:
        plt.figure(figsize=(15, 8))
        crisis_weights.plot(kind='area', stacked=True, colormap='viridis')
        plt.title(f'Evolución de Pesos durante {crisis_name}')
        plt.xlabel('Fecha')
        plt.ylabel('Peso en la Cartera')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'results/crisis_analysis/weights_during_{crisis_name.replace(" ", "_").lower()}.png')

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

for col, format_str in format_dict.items():
    metrics_comparison[col] = metrics_comparison[col].map(lambda x: format_str.format(x))

# Guardar la tabla comparativa
metrics_comparison.to_csv('results/crisis_analysis/metrics_comparison.csv')

# Crear una tabla de correlaciones entre activos durante las crisis
correlation_tables = {}

for crisis_name, (start, end) in crisis_periods.items():
    crisis_data = etf_data.loc[start:end]
    if len(crisis_data) > 0:
        # Calcular rendimientos de los ETFs durante la crisis
        crisis_etf_returns = crisis_data.pct_change().dropna()
        
        # Calcular matriz de correlación
        correlation_matrix = crisis_etf_returns.corr()
        correlation_tables[crisis_name] = correlation_matrix
        
        # Visualizar como heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        plt.title(f'Correlaciones entre ETFs durante {crisis_name}')
        plt.tight_layout()
        plt.savefig(f'results/crisis_analysis/correlations_{crisis_name.replace(" ", "_").lower()}.png')

# Crear un informe resumen en formato de texto
with open('results/crisis_analysis/crisis_analysis_summary.txt', 'w') as f:
    f.write("ANÁLISIS DE CARTERA EQUIPONDERADA DURANTE PERIODOS DE CRISIS\n")
    f.write("============================================================\n\n")
    
    f.write("MÉTRICAS COMPARATIVAS\n")
    f.write("--------------------\n")
    f.write(metrics_comparison.to_string())
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
            
            # Añadir métricas específicas de crisis
            if 'Total Value Change' in metrics:
                f.write(f"Cambio total en valor: {metrics['Total Value Change']:.2%}\n")
            if 'Recovery Days' in metrics:
                f.write(f"Días para recuperación: {metrics['Recovery Days']}\n")
            if 'Avg Safe Asset Weight' in metrics:
                f.write(f"Peso promedio en activos refugio: {metrics['Avg Safe Asset Weight']:.2%}\n")
            
            f.write("\n")
        else:
            f.write("No hay datos suficientes para este periodo.\n\n")
    
    f.write("\nNOTAS ADICIONALES\n")
    f.write("----------------\n")
    f.write("- Los periodos de crisis analizados son:\n")
    for name, (start, end) in crisis_periods.items():
        f.write(f"  * {name}: {start} a {end}\n")
    f.write("\n- El análisis se basa en un portafolio equiponderado de los ETFs: " + ", ".join(selected_etfs) + "\n")
    f.write("- Se aplica rebalanceo mensual con simulación de costos de transacción.\n")
    f.write("- Los activos considerados 'refugio' para el análisis son GLD (oro) y TLT (bonos del tesoro a largo plazo).\n")

# Mostrar resumen en consola
print("\nAnálisis de crisis completado. Resultados guardados en la carpeta 'results/crisis_analysis'.")
print("\nMétricas comparativas:")
print(metrics_comparison)