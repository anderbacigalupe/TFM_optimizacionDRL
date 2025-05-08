import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from datetime import datetime
import sys

# Añadir el directorio raíz del proyecto al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importamos nuestro entorno y agente
from entorno.entorno_cartera import PortfolioEnv
from agentes.agente_dqn import DQNAgent

def load_data(data_path):
    """
    Carga los datos de precios históricos.
    """
    try:
        # Intenta cargar el archivo con la primera columna como índice
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        data = df.values
        asset_names = df.columns.tolist()
        dates = df.index.tolist()
    except:
        try:
            # Intenta cargar el archivo con formato tradicional
            df = pd.read_csv(data_path)
            # Asumimos que la primera columna es la fecha
            data = df.iloc[:, 1:].values
            asset_names = df.columns[1:].tolist()
            dates = df.iloc[:, 0].tolist()
        except Exception as e:
            print(f"Error al cargar los datos: {e}")
            raise
    
    return data, asset_names, dates

def find_latest_model():
    """
    Encuentra el modelo DQN más reciente en la carpeta 'modelos'.
    """
    model_dirs = [d for d in os.listdir('modelos') if d.startswith('dqn_')]
    
    if not model_dirs:
        print("No se encontraron modelos DQN. Por favor, especifica la ruta manualmente.")
        return None
    
    # Ordenamos por fecha (asumiendo formato YYYYMMDD_HHMMSS)
    model_dirs.sort(reverse=True)
    latest_dir = os.path.join('modelos', model_dirs[0])
    
    # Buscamos el mejor modelo o el último
    model_path = os.path.join(latest_dir, 'best_model.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(latest_dir, 'final_model.pth')
    
    if os.path.exists(model_path):
        print(f"Modelo encontrado: {model_path}")
        return model_path
    else:
        print(f"No se encontró un modelo en {latest_dir}")
        return None

def evaluate_agent(agent, env, render=False):
    """
    Evalúa el rendimiento del agente en el entorno.
    """
    state, _ = env.reset()
    done = False
    total_reward = 0
    portfolio_values = [env.balance]
    weights_history = []
    steps = 0
    
    while not done:
        # Seleccionamos la mejor acción según la política actual
        action = agent.select_action(state, training=False)
        
        # Ejecutamos la acción en el entorno
        next_state, reward, done, _, info = env.step(action)
        
        if render:
            env.render()
        
        # Actualizamos el estado y la recompensa
        state = next_state
        total_reward += reward
        
        # Guardamos el valor del portafolio y los pesos
        portfolio_values.append(env.balance)
        weights_history.append(env.portfolio_weights.copy())
        
        steps += 1
    
    return {
        'total_reward': total_reward,
        'final_balance': env.balance,
        'portfolio_values': portfolio_values,
        'weights_history': weights_history,
        'steps': steps
    }

def plot_portfolio_value(portfolio_values, dates, save_path=None):
    """
    Grafica la evolución del valor de la cartera.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values, linewidth=2)
    
    # Si hay muchas fechas, mostramos un subconjunto
    if len(dates) > 50:
        step = len(dates) // 10
        plt.xticks(range(0, len(portfolio_values), step), [dates[min(i, len(dates)-1)] for i in range(0, len(portfolio_values), step)], rotation=45)
    
    plt.xlabel('Fecha')
    plt.ylabel('Valor de la Cartera ($)')
    plt.title('Evolución del Valor de la Cartera - DQN')
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Gráfico guardado en {save_path}")
    
    plt.show()

def calculate_performance_metrics(portfolio_values, risk_free_rate=0.01):
    """
    Calcula métricas de rendimiento del portafolio.
    
    Returns:
        dict: Diccionario con métricas de rendimiento:
            - Rendimiento anualizado
            - Volatilidad anualizada
            - Ratio de Sharpe
            - Ratio de Sortino
            - VaR95 diario
            - Máximo drawdown
    """
    # Convertimos a array de numpy
    portfolio_values = np.array(portfolio_values)
    
    # Calculamos rendimientos diarios
    portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    # Rendimiento total
    total_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
    
    # Rendimiento anualizado (asumiendo 252 días de trading)
    annual_return = ((1 + total_return / 100) ** (252 / len(portfolio_returns)) - 1) * 100
    
    # Volatilidad anualizada
    volatility = np.std(portfolio_returns) * np.sqrt(252) * 100
    
    # Ratio de Sharpe
    excess_return = annual_return / 100 - risk_free_rate
    sharpe_ratio = excess_return / (volatility / 100) if volatility > 0 else 0
    
    # Ratio de Sortino (solo considera la volatilidad negativa)
    negative_returns = portfolio_returns[portfolio_returns < 0]
    downside_deviation = np.std(negative_returns) * np.sqrt(252) * 100 if len(negative_returns) > 0 else 0
    sortino_ratio = excess_return / (downside_deviation / 100) if downside_deviation > 0 else 0
    
    # VaR 95% diario
    var_95 = np.percentile(portfolio_returns, 5) * 100
    
    # Maximum Drawdown
    cumulative_returns = np.cumprod(1 + portfolio_returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = (running_max - cumulative_returns) / running_max
    max_drawdown = np.max(drawdowns) * 100 if len(drawdowns) > 0 else 0
    
    return {
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'var_95': var_95,
        'max_drawdown': max_drawdown
    }

def main():
    # Tratamos de encontrar el modelo más reciente
    model_path = find_latest_model()
    if not model_path:
        model_path = input("Introduce la ruta al modelo DQN a evaluar: ")
    
    # Buscamos el archivo de datos
    data_paths = [
        'data/processed/processed_prices.csv',
        'datos/processed_prices.csv',
        'datos/precios_historicos.csv',
        './processed_prices.csv'
    ]
    
    data_path = None
    for path in data_paths:
        if os.path.exists(path):
            data_path = path
            print(f"Usando datos de: {path}")
            break
            
    if not data_path:
        data_path = input("Introduce la ruta al archivo CSV con los datos de precios: ")
    
    # Creamos directorio para resultados
    results_dir = f"resultados/dqn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Cargamos datos
    data, asset_names, dates = load_data(data_path)
    print(f"Datos cargados con {len(asset_names)} activos: {', '.join(asset_names)}")
    
    # Creamos el entorno
    env = PortfolioEnv(data=data)
    
    # Definimos dimensiones de estados y acciones
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Creamos y cargamos el agente
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        n_discrete_bins=5,  # Usar el mismo número que en el entrenamiento
        min_weight=0.05     # Peso mínimo por activo (5%)
    )
    agent.load(model_path)
    print("Modelo cargado correctamente.")
    
    # Evaluamos el agente
    print("Evaluando agente...")
    eval_results = evaluate_agent(agent, env, render=False)
    print(f"Evaluación completada en {eval_results['steps']} pasos.")
    print(f"Balance final: ${eval_results['final_balance']:.2f}")
    
    # Graficamos la evolución del valor de la cartera
    plot_portfolio_value(
        eval_results['portfolio_values'],
        dates[:len(eval_results['portfolio_values'])],
        os.path.join(results_dir, 'portfolio_value.png')
    )
    
    # Calculamos métricas de rendimiento
    metrics = calculate_performance_metrics(eval_results['portfolio_values'])
    
    # Mostramos métricas
    print("\nMétricas de rendimiento:")
    print(f"Rendimiento anualizado: {metrics['annual_return']:.2f}%")
    print(f"Volatilidad anualizada: {metrics['volatility']:.2f}%")
    print(f"Ratio de Sharpe: {metrics['sharpe_ratio']:.2f}")
    print(f"Ratio de Sortino: {metrics['sortino_ratio']:.2f}")
    print(f"VaR95 diario: {metrics['var_95']:.2f}%")
    print(f"Máximo Drawdown: {metrics['max_drawdown']:.2f}%")
    
    # Guardamos las métricas en un archivo de texto
    with open(os.path.join(results_dir, 'performance_metrics.txt'), 'w') as f:
        f.write("Métricas de rendimiento del modelo DQN\n")
        f.write("======================================\n\n")
        f.write(f"Rendimiento anualizado: {metrics['annual_return']:.4f}%\n")
        f.write(f"Volatilidad anualizada: {metrics['volatility']:.4f}%\n")
        f.write(f"Ratio de Sharpe: {metrics['sharpe_ratio']:.4f}\n")
        f.write(f"Ratio de Sortino: {metrics['sortino_ratio']:.4f}\n")
        f.write(f"VaR95 diario: {metrics['var_95']:.4f}%\n")
        f.write(f"Máximo Drawdown: {metrics['max_drawdown']:.4f}%\n")
    
    print(f"\nResultados guardados en: {results_dir}")

if __name__ == "__main__":
    main()