import pandas as pd
import numpy as np
import cvxpy as cp
from entorno.entorno_cartera import PortfolioEnv
import matplotlib.pyplot as plt

# Cargar datos procesados
prices = pd.read_csv('data/processed/processed_prices.csv', index_col=0, parse_dates=True)

# Eliminar filas con valores NaN (antes de que HYG empiece a cotizar)
prices = prices.dropna()

# Calcular los rendimientos diarios
returns = prices.pct_change().dropna()

# Parámetros de simulación
initial_balance = 1_000_000
rebalance_frequency = 'ME'
first_training_end = pd.Timestamp('2008-01-31')
end_date = pd.Timestamp('2025-04-10')

start_date = returns.index[0]

all_month_ends = pd.date_range(start=first_training_end, end=end_date, freq=rebalance_frequency)
all_month_ends = all_month_ends.union([end_date])
rebalance_dates = [d for d in all_month_ends if d in prices.index]
if rebalance_dates[-1] != end_date and end_date in prices.index:
    rebalance_dates.append(end_date)

portfolio_values = []
portfolio_weights_history = []
portfolio_sharpe_ratios = []
cash_history = []
shares_history = []

risk_free_rate = 0.0

print(f"Iniciando simulación desde {start_date} hasta {end_date}")
print(f"Total de fechas de rebalanceo: {len(rebalance_dates)}")

# Preparar todos los datos de precios para la simulación completa
full_simulation_dates = prices.loc[rebalance_dates[0]:rebalance_dates[-1]].index
full_simulation_prices = prices.loc[full_simulation_dates].values

# Crear una instancia del entorno para toda la simulación
env = PortfolioEnv(full_simulation_prices, initial_balance=initial_balance)
obs, _ = env.reset()

# Guardar el estado inicial
portfolio_values.append((full_simulation_dates[0], initial_balance))
# Guardar el cash inicial
cash_history.append((full_simulation_dates[0], obs[0]))
# Guardar acciones iniciales (todas en cero)
shares_history.append((full_simulation_dates[0], np.zeros(len(prices.columns))))

# Diccionario para mapear fechas a índices en el arreglo de precios
date_to_index = {date: idx for idx, date in enumerate(full_simulation_dates)}

for i in range(len(rebalance_dates) - 1):
    date = rebalance_dates[i]
    next_date = rebalance_dates[i + 1]

    print(f"Procesando periodo {i+1}/{len(rebalance_dates)-1}: {date} a {next_date}")

    # Calcular los nuevos pesos óptimos
    in_sample_returns = returns.loc[start_date:date]
    mu = in_sample_returns.mean() * 252
    sigma = in_sample_returns.cov() * 252

    # Estabilizar la matriz de covarianza
    min_eig = np.min(np.real(np.linalg.eigvals(sigma)))
    if min_eig < 0:
        sigma = sigma + (-min_eig + 1e-8) * np.eye(len(sigma))

    n_assets = len(mu)
    
    try:
        # Método 1: Optimización directa del ratio de Sharpe
        w = cp.Variable(n_assets)
        ret = mu.values @ w
        risk = cp.sqrt(cp.quad_form(w, sigma))
        sharpe_ratio = (ret - risk_free_rate) / risk
            
        # Formular el problema para maximizar el ratio de Sharpe
        problem = cp.Problem(
            cp.Maximize(sharpe_ratio),
            [cp.sum(w) == 1, w >= 0]  # Restricción de suma 1 y no posiciones cortas
        )
            
        problem.solve()
            
        if problem.status == cp.OPTIMAL:
            weights = w.value
            portfolio_return = mu.values @ weights
            portfolio_risk = np.sqrt(weights @ sigma @ weights)
            sharpe = (portfolio_return - risk_free_rate) / portfolio_risk
                
            print(f"  Máximo Sharpe (optimización directa) - Return: {portfolio_return:.4f}, Risk: {portfolio_risk:.4f}, Sharpe: {sharpe:.4f}")
        else:
            raise ValueError("No se pudo resolver el problema de optimización directa")
                
    except Exception as e2:
        print(f"  No se pudo usar optimización directa: {str(e2)}")
            
        try:
            # Método 2: Enfoque iterativo buscando entre múltiples carteras
            best_sharpe = -np.inf
            best_weights = None
                
            # Primero encontrar el portafolio de mínima varianza como base
            w_min_var = cp.Variable(n_assets)
            min_var_problem = cp.Problem(
                cp.Minimize(cp.quad_form(w_min_var, sigma)),
                [cp.sum(w_min_var) == 1, w_min_var >= 0]
            )
                
            min_var_problem.solve()
                
            if min_var_problem.status != cp.OPTIMAL:
                raise ValueError("No se pudo encontrar el portafolio de mínima varianza")
                
            min_var_weights = w_min_var.value
            min_return = mu.values @ min_var_weights

            # Validación robusta para evitar valores negativos o rangos inválidos
            if mu.max() < 0:
                raise ValueError("Todos los retornos esperados son negativos; no se puede generar portafolios eficientes.")
                
            lower_bound = max(0.0001, min_return)
            upper_bound = max(lower_bound + 0.0001, mu.max())
                
            # Buscar portafolios con retornos mayores al de mínima varianza
            # Asegurar que los targets sean estrictamente positivos
            target_returns = np.linspace(lower_bound, upper_bound, 50)
                
            for target_ret in target_returns:
                w_target = cp.Variable(n_assets)
                target_problem = cp.Problem(
                    cp.Minimize(cp.quad_form(w_target, sigma)),
                    [mu.values @ w_target >= target_ret, cp.sum(w_target) == 1, w_target >= 0]
                )
                    
                try:
                    target_problem.solve()
                        
                    if target_problem.status == cp.OPTIMAL:
                        w_candidate = w_target.value
                        ret_candidate = mu.values @ w_candidate
                        risk_candidate = np.sqrt(w_candidate @ sigma @ w_candidate)
                        sharpe_candidate = (ret_candidate - risk_free_rate) / risk_candidate
                            
                        if sharpe_candidate > best_sharpe:
                            best_sharpe = sharpe_candidate
                            best_weights = w_candidate
                except:
                    continue
                
            if best_weights is not None:
                weights = best_weights
                portfolio_return = mu.values @ weights
                portfolio_risk = np.sqrt(weights @ sigma @ weights)
                sharpe = best_sharpe
                    
                print(f"  Máximo Sharpe (método iterativo) - Return: {portfolio_return:.4f}, Risk: {portfolio_risk:.4f}, Sharpe: {sharpe:.4f}")
            else:
                raise ValueError("No se encontró ningún portafolio válido en el método iterativo")
            
        except Exception as e3:
            print(f"  Todos los métodos fallaron. Usando pesos equiponderados: {str(e3)}")
            weights = np.array([1/n_assets] * n_assets)
            portfolio_return = mu.values @ weights
            portfolio_risk = np.sqrt(weights @ sigma @ weights)
            sharpe = (portfolio_return - risk_free_rate) / portfolio_risk
            print(f"  Portafolio equiponderado - Return: {portfolio_return:.4f}, Risk: {portfolio_risk:.4f}, Sharpe: {sharpe:.4f}")

    # Guardar los datos
    portfolio_sharpe_ratios.append((date, sharpe))
    portfolio_weights_history.append((date, weights))

    # Asegurar que los pesos son válidos
    weights = np.clip(weights, 0, None)
    weights = weights / np.sum(weights)

    # Simular hasta la siguiente fecha de rebalanceo
    current_index = date_to_index[date]
    next_index = date_to_index[next_date]
    
    # La primera fecha ya se ha simulado en el paso anterior o al inicio
    steps_to_simulate = next_index - current_index
    
    # Aplicar los pesos y simular hasta la próxima fecha de rebalanceo
    for _ in range(steps_to_simulate):
        obs, reward, done, truncated, info = env.step(weights)
        if done:
            break
    
    # Guardar el valor del portafolio después del periodo
    portfolio_value = info["portfolio_value"]  # Usar el valor total del portafolio del info
    cash = info["cash"]  # Guardar el efectivo disponible
    shares = info["shares"]  # Guardar las acciones
    
    portfolio_values.append((next_date, portfolio_value))
    cash_history.append((next_date, cash))
    shares_history.append((next_date, shares))
    
    print(f"  Valor del portfolio al {next_date}: ${portfolio_value:.2f}")
    print(f"  Efectivo disponible: ${cash:.2f} ({cash/portfolio_value*100:.2f}%)")

# Guardar datos en CSV
values_df = pd.DataFrame(portfolio_values, columns=['date', 'portfolio_value']).set_index('date')
weights_df = pd.DataFrame(
    [w for _, w in portfolio_weights_history],
    index=[d for d, _ in portfolio_weights_history],
    columns=prices.columns, dtype='float64'
)
cash_df = pd.DataFrame(cash_history, columns=['date', 'cash']).set_index('date')

# Guardar historial de acciones
shares_df_list = []
for date, shares in shares_history:
    shares_dict = {col: shares[i] for i, col in enumerate(prices.columns)}
    shares_dict['date'] = date
    shares_df_list.append(shares_dict)
shares_df = pd.DataFrame(shares_df_list).set_index('date')

values_df.to_csv('results/markowitz_portfolio_values.csv')
weights_df.to_csv('results/markowitz_weights.csv')
cash_df.to_csv('results/markowitz_cash.csv')
shares_df.to_csv('results/markowitz_shares.csv')

# --- CÁLCULO DE MÉTRICAS ---

def calcular_metricas(serie_valores):
    returns = serie_valores.pct_change().dropna()
    log_returns = np.log1p(returns)
    mean_daily = returns.mean()
    std_daily = returns.std()
    downside_std = returns[returns < 0].std()
    annual_return = (serie_valores.iloc[-1] / serie_valores.iloc[0]) ** (252 / len(returns)) - 1
    annual_volatility = std_daily * np.sqrt(252)
    sharpe = (mean_daily * 252) / (std_daily * np.sqrt(252))
    sortino = (mean_daily * 252) / (downside_std * np.sqrt(252)) if downside_std > 0 else np.nan
    rolling_max = serie_valores.cummax()
    drawdowns = (serie_valores - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    var_95 = np.percentile(returns, 5)
    return {
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown_pct': max_drawdown,
        'VaR_95': var_95
    }

# Calcular métricas del portfolio Markowitz
markowitz_metrics = calcular_metricas(values_df['portfolio_value'])

# Calcular métricas del portfolio equiponderado
equal_weights = np.array([1 / len(prices.columns)] * len(prices.columns))

# Crear un nuevo entorno para el portafolio equiponderado utilizando los mismos datos
env_eq = PortfolioEnv(full_simulation_prices, initial_balance=initial_balance)
obs_eq, _ = env_eq.reset()

portfolio_eq_values = [(full_simulation_dates[0], initial_balance)]

# Simulación del portafolio equiponderado
for i in range(len(rebalance_dates) - 1):
    date = rebalance_dates[i]
    next_date = rebalance_dates[i + 1]
    
    current_index = date_to_index[date]
    next_index = date_to_index[next_date]
    
    steps_to_simulate = next_index - current_index
    
    # La primera fecha ya se ha simulado en el reset
    if i > 0:
        # Aplicar los pesos y simular hasta la próxima fecha de rebalanceo
        for _ in range(steps_to_simulate):
            obs_eq, reward_eq, done_eq, truncated_eq, info_eq = env_eq.step(equal_weights)
            if done_eq:
                break
        
        # Guardar el valor del portafolio después del periodo
        portfolio_eq_value = info_eq["portfolio_value"]
        portfolio_eq_values.append((next_date, portfolio_eq_value))
    else:
        # Para el primer paso, simplemente aplicamos los pesos
        obs_eq, reward_eq, done_eq, truncated_eq, info_eq = env_eq.step(equal_weights)
        portfolio_eq_value = info_eq["portfolio_value"]
        portfolio_eq_values.append((date, portfolio_eq_value))

values_eq_df = pd.DataFrame(portfolio_eq_values, columns=['date', 'portfolio_value']).set_index('date')
equal_metrics = calcular_metricas(values_eq_df['portfolio_value'])

# Guardar métricas
metrics_df = pd.DataFrame([markowitz_metrics, equal_metrics], index=['markowitz', 'equiponderado'])
metrics_df.to_csv('results/portfolio_metrics.csv')

# --- VISUALIZACIÓN ---

# Evolución del valor del portfolio
plt.figure(figsize=(12, 6))
plt.plot(values_df.index, values_df['portfolio_value'], label='Markowitz')
plt.plot(values_eq_df.index, values_eq_df['portfolio_value'], label='Equiponderado', linestyle='--')
plt.title('Evolución del valor del portfolio')
plt.xlabel('Fecha')
plt.ylabel('Valor del portfolio')
plt.legend()
plt.grid(True)
plt.savefig('figures/markowitz_vs_equal.png')
plt.close()

# Evolución de los pesos del portfolio
weights_df.plot.area(stacked=True, figsize=(12, 6), title='Evolución de los pesos del portfolio')
plt.xlabel('Fecha')
plt.ylabel('Peso en cartera')
plt.legend(loc='upper left')
plt.tight_layout()
plt.grid(True)
plt.savefig('figures/pesos_markowitz.png')
plt.close()

# Mostrar resumen de métricas
print("\n--- RESUMEN DE RENDIMIENTO ---")
print(metrics_df.round(4))
print("\nResultados guardados en las carpetas 'results' y 'figures'")
