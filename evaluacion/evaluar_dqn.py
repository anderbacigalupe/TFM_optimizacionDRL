import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from datetime import datetime
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

# Añadir el directorio raíz del proyecto al path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Importamos nuestro entorno y agente
from entorno.entorno_cartera import PortfolioEnv
from agentes.agente_dqn import DQNAgent


def load_data(data_path: str) -> Tuple[np.ndarray, List[str], List[Any]]:
    """
    Carga los datos de precios históricos de manera eficiente.
    
    Args:
        data_path: Ruta al archivo CSV con los datos de precios
        
    Returns:
        Tupla con (datos, nombres de activos, fechas)
    """
    try:
        # La opción más común primero para evitar excepciones
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        return df.values, df.columns.tolist(), df.index.tolist()
    except Exception as e:
        try:
            # Formato alternativo
            df = pd.read_csv(data_path)
            # Asumimos que la primera columna es la fecha
            return df.iloc[:, 1:].values, df.columns[1:].tolist(), df.iloc[:, 0].tolist()
        except Exception as nested_e:
            print(f"Error al cargar los datos: {e}\nDetalle adicional: {nested_e}")
            raise


def find_model_path(model_dir: str = 'modelos', prefix: str = 'dqn_') -> Optional[str]:
    """
    Encuentra el modelo más reciente o pide input al usuario.
    
    Args:
        model_dir: Directorio donde buscar modelos
        prefix: Prefijo de los directorios de modelos
        
    Returns:
        Ruta al modelo o None si no se encuentra
    """
    if not os.path.exists(model_dir):
        print(f"El directorio {model_dir} no existe.")
        return None
    
    model_dirs = [d for d in os.listdir(model_dir) if d.startswith(prefix)]
    
    if not model_dirs:
        return None
    
    # Ordenamos por fecha (asumiendo formato YYYYMMDD_HHMMSS)
    model_dirs.sort(reverse=True)
    latest_dir = os.path.join(model_dir, model_dirs[0])
    
    # Buscamos primero el mejor modelo, luego el final
    for model_name in ['best_model.pth', 'final_model.pth']:
        model_path = os.path.join(latest_dir, model_name)
        if os.path.exists(model_path):
            print(f"Modelo encontrado: {model_path}")
            return model_path
    
    return None


def evaluate_agent(agent: DQNAgent, env: PortfolioEnv, render: bool = False) -> Dict[str, Any]:
    """
    Evalúa el rendimiento del agente en el entorno de manera eficiente.
    
    Args:
        agent: Agente DQN a evaluar
        env: Entorno de portafolio
        render: Si se debe renderizar el entorno
        
    Returns:
        Diccionario con métricas y datos de la evaluación
    """
    state, _ = env.reset()
    done = False
    
    # Preallocate arrays for better performance
    max_steps = 10000  # Maximum possible steps (adjust based on data)
    portfolio_values = np.zeros(max_steps + 1)
    portfolio_values[0] = env.balance
    daily_returns = np.zeros(max_steps)
    
    weights_history = []
    step = 0
    total_reward = 0
    
    # Valor anterior para cálculo de retornos
    previous_value = env.balance
    
    while not done and step < max_steps:
        # Seleccionamos la mejor acción (sin exploración)
        action = agent.select_action(state, training=False)
        
        # Ejecutamos la acción en el entorno
        next_state, reward, done, _, info = env.step(action)
        
        if render and step % 20 == 0:  # Reducir frecuencia de renderizado para eficiencia
            env.render()
        
        # Calcular retorno diario
        current_value = info.get('portfolio_value', env.balance)
        daily_returns[step] = current_value / previous_value - 1
        previous_value = current_value
        
        # Actualizamos estado y recompensa
        state = next_state
        total_reward += reward
        
        # Guardamos valor del portafolio y pesos
        step += 1
        portfolio_values[step] = env.balance
        weights_history.append(env.portfolio_weights.copy())
    
    # Recortar arrays al tamaño real usado
    portfolio_values = portfolio_values[:step+1]
    daily_returns = daily_returns[:step]
    
    # Calcular métricas
    metrics = calculate_performance_metrics(portfolio_values, daily_returns)
    
    return {
        'total_reward': total_reward,
        'final_balance': env.balance,
        'portfolio_values': portfolio_values.tolist(),  # Convertir a lista para serialización
        'weights_history': weights_history,
        'steps': step,
        'daily_returns': daily_returns.tolist(),  # Convertir a lista para serialización
        **metrics  # Incluir todas las métricas calculadas
    }


def calculate_performance_metrics(
    portfolio_values: np.ndarray, 
    daily_returns: np.ndarray, 
    risk_free_rate: float = 0.00,
    trading_days: int = 252
) -> Dict[str, float]:
    """
    Calcula métricas de rendimiento del portafolio de manera eficiente.
    
    Args:
        portfolio_values: Array con los valores de la cartera
        daily_returns: Array con los retornos diarios
        risk_free_rate: Tasa libre de riesgo anualizada
        trading_days: Días de trading por año
        
    Returns:
        Diccionario con métricas de rendimiento
    """
    # Filtrar valores NaN o cero en returns
    valid_returns = daily_returns[~np.isnan(daily_returns)]
    
    if len(valid_returns) == 0:
        return {
            'annual_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'var_95': 0.0,
            'max_drawdown': 0.0
        }
    
    # Rendimiento total y anualizado
    total_return = (portfolio_values[-1] / portfolio_values[0] - 1)
    annual_return = ((1 + total_return) ** (trading_days / len(valid_returns)) - 1) * 100
    
    # Volatilidad anualizada
    volatility = np.std(valid_returns) * np.sqrt(trading_days) * 100
    
    # Ratio de Sharpe
    excess_return = annual_return / 100 - risk_free_rate
    sharpe_ratio = excess_return / (volatility / 100) if volatility > 0 else 0
    
    # Ratio de Sortino (solo volatilidad negativa)
    negative_returns = valid_returns[valid_returns < 0]
    downside_deviation = np.std(negative_returns) * np.sqrt(trading_days) * 100 if len(negative_returns) > 0 else 1e-6
    sortino_ratio = excess_return / (downside_deviation / 100)
    
    # VaR 95% diario
    var_95 = np.percentile(valid_returns, 5) * 100
    
    # Maximum Drawdown (cálculo vectorizado)
    cumulative_returns = np.cumprod(1 + valid_returns)
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


def plot_portfolio_value(
    portfolio_values: List[float], 
    dates: List[Any], 
    title: str = 'Evolución del Valor de la Cartera - DQN',
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Grafica la evolución del valor de la cartera con mejor formato.
    
    Args:
        portfolio_values: Lista o array con los valores de la cartera
        dates: Lista de fechas
        title: Título del gráfico
        save_path: Ruta para guardar el gráfico (opcional)
        show_plot: Si se debe mostrar el gráfico
    """
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values, linewidth=2, color='#1f77b4')
    
    # Definir sensatamente los ticks del eje X
    max_ticks = 10
    if len(dates) > max_ticks:
        step = len(dates) // max_ticks
        indices = range(0, len(portfolio_values), step)
        tick_dates = [dates[min(i, len(dates)-1)] for i in indices]
        
        plt.xticks(indices, tick_dates, rotation=45)
    
    # Mejorar formato y estilo
    plt.xlabel('Fecha', fontsize=12)
    plt.ylabel('Valor de la Cartera ($)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Añadir valor inicial y final para referencia
    plt.annotate(f'Inicial: ${portfolio_values[0]:,.2f}', 
                xy=(0, portfolio_values[0]),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=10)
    
    plt.annotate(f'Final: ${portfolio_values[-1]:,.2f}', 
                xy=(len(portfolio_values)-1, portfolio_values[-1]),
                xytext=(-70, 10),
                textcoords='offset points',
                fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico guardado en {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def find_data_path() -> str:
    """Encuentra la ruta a los datos de precios"""
    data_paths = [
        'data/processed/processed_prices.csv',
        'datos/processed_prices.csv',
        'datos/precios_historicos.csv',
        './processed_prices.csv',
        os.path.join(project_root, 'data/processed/processed_prices.csv')
    ]
    
    for path in data_paths:
        if os.path.exists(path):
            print(f"Usando datos de: {path}")
            return path
            
    return input("Introduce la ruta al archivo CSV con los datos de precios: ")


def save_results(results: Dict[str, Any], results_dir: str) -> None:
    """
    Guarda los resultados en archivos.
    
    Args:
        results: Diccionario con los resultados de la evaluación
        results_dir: Directorio donde guardar los resultados
    """
    # Asegurar que el directorio existe
    os.makedirs(results_dir, exist_ok=True)
    
    # Guardar métricas en formato texto
    metrics_path = os.path.join(results_dir, 'performance_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("Métricas de rendimiento del modelo DQN\n")
        f.write("======================================\n\n")
        f.write(f"Rendimiento anualizado: {results['annual_return']:.4f}%\n")
        f.write(f"Volatilidad anualizada: {results['volatility']:.4f}%\n")
        f.write(f"Ratio de Sharpe: {results['sharpe_ratio']:.4f}\n")
        f.write(f"Ratio de Sortino: {results['sortino_ratio']:.4f}\n")
        f.write(f"VaR95 diario: {results['var_95']:.4f}%\n")
        f.write(f"Máximo Drawdown: {results['max_drawdown']:.4f}%\n")
        f.write(f"\nBalance final: ${results['final_balance']:.2f}\n")
        f.write(f"Recompensa total: {results['total_reward']:.4f}\n")
        f.write(f"Pasos totales: {results['steps']}\n")
    
    # Guardar valores del portafolio en CSV
    portfolio_df = pd.DataFrame({
        'Value': results['portfolio_values'],
        'Return': [0] + results['daily_returns']  # Añadir 0 al inicio para alinear
    })
    portfolio_df.to_csv(os.path.join(results_dir, 'portfolio_values.csv'), index=False)
    
    print(f"Resultados guardados en: {results_dir}")


def main() -> None:
    """Función principal del evaluador de DQN"""
    # Configurar manejo de excepciones para depuración robusta
    try:
        # Encontrar y cargar modelo
        model_path = find_model_path()
        if not model_path:
            model_path = input("Introduce la ruta al modelo DQN a evaluar: ")
        
        # Encontrar datos
        data_path = find_data_path()
        
        # Crear directorio para resultados
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join("results", f"dqn_eval_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)
        
        # Cargar datos
        data, asset_names, dates = load_data(data_path)
        print(f"Datos cargados: {len(dates)} días, {len(asset_names)} activos")
        
        # Crear entorno
        env = PortfolioEnv(data=data)
        
        # Dimensiones de estados y acciones
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        # Crear y cargar agente
        print("Cargando agente DQN...")
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            n_discrete_bins=5,
            min_weight=0.05
        )
        agent.load(model_path)
        print("Modelo cargado correctamente.")
        
        # Evaluar agente
        print("Evaluando agente DQN...")
        start_time = datetime.now()
        eval_results = evaluate_agent(agent, env, render=False)
        eval_time = (datetime.now() - start_time).total_seconds()
        print(f"Evaluación completada en {eval_time:.2f} segundos ({eval_results['steps']} pasos)")
        
        # Mostrar resumen de resultados
        print("\n==== RESUMEN DE RESULTADOS ====")
        print(f"Balance final: ${eval_results['final_balance']:,.2f}")
        print(f"Rendimiento anualizado: {eval_results['annual_return']:.2f}%")
        print(f"Volatilidad anualizada: {eval_results['volatility']:.2f}%")
        print(f"Ratio de Sharpe: {eval_results['sharpe_ratio']:.2f}")
        print(f"Ratio de Sortino: {eval_results['sortino_ratio']:.2f}")
        print(f"VaR95 diario: {eval_results['var_95']:.2f}%")
        print(f"Máximo Drawdown: {eval_results['max_drawdown']:.2f}%")
        
        # Generar gráficos
        plot_portfolio_value(
            eval_results['portfolio_values'],
            dates[:len(eval_results['portfolio_values'])],
            save_path=os.path.join(results_dir, 'portfolio_value.png')
        )
        
        # Guardar resultados
        save_results(eval_results, results_dir)
        
    except Exception as e:
        print(f"\nERROR durante la evaluación: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())