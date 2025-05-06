import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


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
    
    # Aplicar slippage al precio (compras a precio más alto, ventas a precio más bajo)
    adjusted_prices_buy = prices * (1 + slippage_percentage)
    adjusted_prices_sell = prices * (1 - slippage_percentage)
    
    # Separar compras y ventas
    buy_shares = np.maximum(shares_to_trade, 0)
    sell_shares = np.maximum(-shares_to_trade, 0)
    
    # Costo total por slippage
    slippage_costs = sum(buy_shares * prices * slippage_percentage) + sum(sell_shares * prices * slippage_percentage)
    
    # Costos de comisión (considerando mínimo por operación y máximo como porcentaje)
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

# Inicialización del seguimiento de cartera
portfolio_values = []
current_shares = None
available_cash = initial_capital
transaction_costs_history = []
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
    
    # Valor actual de la cartera
    portfolio_value = sum(current_shares * prices) + available_cash
    portfolio_values.append(portfolio_value)

# Crear un DataFrame con los valores del portafolio
portfolio_df = pd.DataFrame({'Portfolio Value': portfolio_values}, index=etf_data.index)

# Calcular rendimientos diarios
portfolio_df['Daily Return'] = portfolio_df['Portfolio Value'].pct_change()

# Eliminar el primer valor (NaN)
portfolio_returns = portfolio_df['Daily Return'].dropna()

# Calcular métricas de rendimiento
# 1. Rentabilidad anualizada
total_days = (portfolio_df.index[-1] - portfolio_df.index[0]).days
years = total_days / 365.25
total_return = (portfolio_df['Portfolio Value'].iloc[-1] / portfolio_df['Portfolio Value'].iloc[0]) - 1
annualized_return = (1 + total_return) ** (1 / years) - 1

# 2. Volatilidad anualizada
daily_volatility = portfolio_returns.std()
annualized_volatility = daily_volatility * np.sqrt(trading_days_per_year)

# 3. Ratio de Sharpe 
risk_free_rate = 0.00  # Aqui se deberia indicar la tasa libre de riesgo, pero para comparar con otras estrategias lo importante es que siempre eligas la misma
sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility

# 4. Ratio de Sortino (solo considera volatilidad a la baja)
negative_returns = portfolio_returns[portfolio_returns < 0]
downside_deviation = negative_returns.std() * np.sqrt(trading_days_per_year)
sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation if len(negative_returns) > 0 else np.nan

# 5. VaR 95%
var_95 = np.percentile(portfolio_returns, 5)
var_95_dollar = var_95 * portfolio_df['Portfolio Value'].iloc[-1]

# 6. Máximo drawdown
cumulative_returns = (1 + portfolio_returns).cumprod()
cumulative_max = cumulative_returns.cummax()
drawdowns = (cumulative_returns / cumulative_max) - 1
max_drawdown = drawdowns.min()

# Crear un DataFrame con las métricas solicitadas
metrics = {
    'Métrica': [
        'Rentabilidad Anualizada', 
        'Volatilidad Anualizada', 
        'Ratio Sharpe', 
        'Ratio Sortino',
        'VaR 95%', 
        'Máximo Drawdown'
    ],
    'Valor': [
        f"{annualized_return:.2%}", 
        f"{annualized_volatility:.2%}", 
        f"{sharpe_ratio:.4f}", 
        f"{sortino_ratio:.4f}",
        f"{var_95:.2%}", 
        f"{max_drawdown:.2%}"
    ]
}
metrics_df = pd.DataFrame(metrics)

# Añadir información adicional relevante
additional_info = {
    'Métrica': [
        'Capital Inicial', 
        'Valor Final', 
        'Retorno Total',
        'Años', 
        'Costos de Transacción Total',
        'Porcentaje Invertido al Final'
    ],
    'Valor': [
        f"${initial_capital:,.2f}", 
        f"${portfolio_df['Portfolio Value'].iloc[-1]:,.2f}", 
        f"{total_return:.2%}",
        f"{years:.2f}", 
        f"${sum(transaction_costs_history):,.2f}",
        f"{(1 - available_cash/portfolio_df['Portfolio Value'].iloc[-1]):.2%}"
    ]
}
additional_info_df = pd.DataFrame(additional_info)

# Guardar las métricas en un CSV
metrics_df.to_csv('results/equalweight_portfolio_metrics.csv')
additional_info_df.to_csv('results/additional_info.csv')

# Guardar también los datos completos para referencia
portfolio_df.to_csv('results/equalweight_portfolio_history.csv')

# Mostrar las métricas en consola
print("\nMétricas de rendimiento:")
print(metrics_df.to_string(index=False))
print("\nInformación adicional:")
print(additional_info_df.to_string(index=False))

# Generar y guardar el gráfico de evolución del valor de la cartera
plt.figure(figsize=(12, 6))
portfolio_df['Portfolio Value'].plot(title='Evolución del Valor de la Cartera', grid=True)
plt.ylabel('USD')
plt.xlabel('Fecha')
plt.tight_layout()
plt.savefig('results/equalweight_portfolio_value.png')

# Número de rebalanceos = número de operaciones (uno por mes)
num_operaciones = len(transaction_costs_history)
total_comisiones = sum(transaction_costs_history)

print(f"\nNúmero total de operaciones (rebalanceos): {num_operaciones}")
print(f"Total de comisiones incurridas: ${total_comisiones:,.2f}")


