import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from datetime import datetime
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import seaborn as sns

# Añadir el directorio raíz del proyecto al path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Importamos nuestro entorno y agente
from entorno.entorno_cartera import PortfolioEnv
from agentes.agente_dqn import DQNAgent

# Define los periodos de crisis
crisis_periods = {
    'Crisis Financiera Global': (pd.to_datetime('2007-10-01'), pd.to_datetime('2009-03-31')),
    'Crisis COVID-19': (pd.to_datetime('2020-02-01'), pd.to_datetime('2020-04-30')),
    'Crisis Inflacionaria': (pd.to_datetime('2021-12-01'), pd.to_datetime('2022-10-31'))
}

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


def evaluate_agent(agent: DQNAgent, env: PortfolioEnv, dates: List[Any], 
                   render: bool = False) -> Dict[str, Any]:
    """
    Evalúa el rendimiento del agente en el entorno de manera eficiente.
    
    Args:
        agent: Agente DQN a evaluar
        env: Entorno de portafolio
        dates: Lista de fechas para el periodo de evaluación
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
    
    # Verificar si tenemos suficientes fechas para los datos generados
    if len(dates) < len(portfolio_values):
        print(f"Advertencia: No hay suficientes fechas ({len(dates)}) para los valores del portafolio ({len(portfolio_values)})")
        # Ajustar el tamaño de portfolio_values para que coincida con las fechas disponibles
        portfolio_values = portfolio_values[:len(dates)]
        daily_returns = daily_returns[:len(dates)-1] if len(daily_returns) > 0 else daily_returns
    
    # Crear DataFrame con fechas (asegurándonos de que las longitudes coincidan)
    eval_dates = dates[:len(portfolio_values)]
    portfolio_df = pd.DataFrame({
        'Value': portfolio_values,
        'Return': np.concatenate(([0], daily_returns[:len(portfolio_values)-1]))  # Añadir 0 al inicio para alinear con valores
    }, index=eval_dates)
    
    # Convertir pesos a DataFrame con fechas (asegurando que las longitudes coincidan)
    if len(weights_history) > 0:
        weight_dates = eval_dates[1:min(len(eval_dates), len(weights_history)+1)]  # Los pesos empiezan desde el segundo día
        weights_df = pd.DataFrame(
            weights_history[:len(weight_dates)],  # Limitamos a las fechas disponibles
            index=weight_dates
        )
    else:
        weights_df = pd.DataFrame()
    
    # Calcular métricas
    metrics = calculate_performance_metrics(portfolio_values, daily_returns[:len(portfolio_values)-1])
    
    return {
        'total_reward': total_reward,
        'final_balance': env.balance,
        'portfolio_df': portfolio_df,
        'weights_df': weights_df,
        'steps': step,
        'metrics': metrics  # Incluir todas las métricas calculadas
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
            'cvar_95': 0.0,
            'max_drawdown': 0.0,
            'total_return': 0.0
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
    
    # VaR 95% diario y CVaR
    var_95 = np.percentile(valid_returns, 5) * 100
    cvar_95 = np.mean(valid_returns[valid_returns <= np.percentile(valid_returns, 5)]) * 100 if any(valid_returns <= np.percentile(valid_returns, 5)) else var_95
    
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
        'cvar_95': cvar_95,
        'max_drawdown': max_drawdown,
        'total_return': total_return * 100  # Convertir a porcentaje
    }


def plot_portfolio_value(
    portfolio_df: pd.DataFrame, 
    title: str = 'Evolución del Valor de la Cartera - DQN',
    highlight_crisis: bool = True,
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Grafica la evolución del valor de la cartera con mejor formato.
    
    Args:
        portfolio_df: DataFrame con valores de la cartera
        title: Título del gráfico
        highlight_crisis: Si se deben resaltar los periodos de crisis
        save_path: Ruta para guardar el gráfico (opcional)
        show_plot: Si se debe mostrar el gráfico
    """
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_df.index, portfolio_df['Value'], linewidth=2, color='#1f77b4')
    
    # Resaltar periodos de crisis si se solicita
    if highlight_crisis:
        # Definir colores para cada crisis
        colors = ['red', 'orange', 'purple']
        for i, (crisis_name, (start, end)) in enumerate(crisis_periods.items()):
            plt.axvspan(start, end, color=colors[i], alpha=0.2, label=f'Periodo de {crisis_name}')
    
    # Mejorar formato y estilo
    plt.xlabel('Fecha', fontsize=12)
    plt.ylabel('Valor de la Cartera ($)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Añadir valor inicial y final para referencia
    plt.annotate(f'Inicial: ${portfolio_df["Value"].iloc[0]:,.2f}', 
                xy=(portfolio_df.index[0], portfolio_df["Value"].iloc[0]),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=10)
    
    plt.annotate(f'Final: ${portfolio_df["Value"].iloc[-1]:,.2f}', 
                xy=(portfolio_df.index[-1], portfolio_df["Value"].iloc[-1]),
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


def analyze_crisis_periods(portfolio_df: pd.DataFrame, weights_df: pd.DataFrame, 
                           asset_names: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Analiza el comportamiento de la cartera durante los períodos de crisis.
    
    Args:
        portfolio_df: DataFrame con valores y retornos de la cartera
        weights_df: DataFrame con los pesos de cada activo
        asset_names: Nombres de los activos
    
    Returns:
        Diccionario con análisis para cada período de crisis
    """
    crisis_analysis = {}
    
    for crisis_name, (start, end) in crisis_periods.items():
        # Convertir fechas a datetime si son strings
        start_date = pd.to_datetime(start) if isinstance(start, str) else start
        end_date = pd.to_datetime(end) if isinstance(end, str) else end
        
        # Verificar si tenemos datos para este periodo (con diagnóstico)
        dates_in_range = portfolio_df.index[(portfolio_df.index >= start_date) & (portfolio_df.index <= end_date)]
        if len(dates_in_range) == 0:
            print(f"No se encontraron datos para el periodo {crisis_name} ({start} a {end})")
            # Incluir fechas más cercanas al rango solicitado
            closest_before = portfolio_df.index[portfolio_df.index < start_date][-5:] if any(portfolio_df.index < start_date) else []
            closest_after = portfolio_df.index[portfolio_df.index > end_date][:5] if any(portfolio_df.index > end_date) else []
            if len(closest_before) > 0 or len(closest_after) > 0:
                print(f"Fechas más cercanas disponibles: {closest_before.tolist()} ... {closest_after.tolist()}")
            continue
        
        # Filtrar datos para el periodo de crisis (usando los índices encontrados)
        crisis_data = portfolio_df.loc[dates_in_range]
        
        print(f"Analizando {crisis_name}: {len(crisis_data)} días de datos encontrados")
        
        if len(crisis_data) > 5:  # Asegurar suficientes datos
            # Calcular métricas para el periodo
            crisis_returns = crisis_data['Return'].values[1:] if len(crisis_data) > 1 else np.array([])  # Eliminar el primer valor (cero)
            crisis_values = crisis_data['Value'].values
            
            # Calcular métricas
            metrics = calculate_performance_metrics(crisis_values, crisis_returns)
            
            # Calcular drawdown
            drawdowns = np.array([])
            if len(crisis_returns) > 0:
                cumulative_returns = np.cumprod(1 + crisis_returns)
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdowns = (running_max - cumulative_returns) / running_max
            
            # Análisis de pesos durante la crisis
            crisis_weights = weights_df.loc[weights_df.index.to_series().between(start_date, end_date)] if not weights_df.empty else pd.DataFrame()
            
            # Días extremos
            worst_days = []
            best_days = []
            if len(crisis_returns) > 0:
                sorted_returns_idx = np.argsort(crisis_returns)
                worst_days_idx = sorted_returns_idx[:min(5, len(sorted_returns_idx))]
                best_days_idx = sorted_returns_idx[max(0, len(sorted_returns_idx)-5):]
                
                # Obtener fechas de días extremos
                if len(crisis_data.index) > max(np.max(worst_days_idx) if len(worst_days_idx) > 0 else 0, 
                                             np.max(best_days_idx) if len(best_days_idx) > 0 else 0) + 1:
                    worst_days = [(crisis_data.index[i+1], crisis_returns[i]) for i in worst_days_idx]
                    best_days = [(crisis_data.index[i+1], crisis_returns[i]) for i in best_days_idx]
            
            # Guardar análisis
            crisis_analysis[crisis_name] = {
                'metrics': metrics,
                'data': crisis_data,
                'weights': crisis_weights,
                'drawdowns': pd.Series(drawdowns, index=crisis_data.index[1:] if len(crisis_data) > 1 else []),
                'worst_days': worst_days,
                'best_days': best_days
            }
        else:
            print(f"Insuficientes datos para {crisis_name}: solo {len(crisis_data)} días disponibles")
    
    return crisis_analysis


def plot_crisis_analysis(crisis_analysis: Dict[str, Dict[str, Any]], results_dir: str) -> None:
    """
    Genera gráficos de análisis para cada período de crisis.
    
    Args:
        crisis_analysis: Resultados del análisis de crisis
        results_dir: Directorio donde guardar los gráficos
    """
    # Directorio para gráficos de crisis
    crisis_dir = os.path.join(results_dir, 'crisis_analysis')
    os.makedirs(crisis_dir, exist_ok=True)
    
    # Colores para gráficos
    colors = ['red', 'orange', 'purple']
    
    # 1. Gráficos de Drawdown por crisis
    plt.figure(figsize=(15, 12))
    for i, (crisis_name, analysis) in enumerate(crisis_analysis.items()):
        if not analysis['drawdowns'].empty:
            plt.subplot(3, 1, i+1)
            analysis['drawdowns'].plot(color=colors[i])
            plt.title(f'Drawdown durante {crisis_name}')
            plt.xlabel('Fecha')
            plt.ylabel('Drawdown (%)')
            plt.grid(True)
            plt.ylim(-1, 0.1)  # Limitamos el eje Y para mejor visualización
    
    plt.tight_layout()
    plt.savefig(os.path.join(crisis_dir, 'drawdowns_during_crisis.png'))
    plt.close()
    
    # 2. Distribución de rendimientos por crisis
    plt.figure(figsize=(15, 12))
    for i, (crisis_name, analysis) in enumerate(crisis_analysis.items()):
        if len(analysis['data']) > 1:  # Asegurar suficientes datos
            plt.subplot(3, 1, i+1)
            returns = analysis['data']['Return'].iloc[1:]  # Eliminar el primer valor (cero)
            sns.histplot(returns, kde=True, color=colors[i])
            plt.axvline(0, color='black', linestyle='--')
            plt.title(f'Distribución de Rendimientos durante {crisis_name}')
            plt.xlabel('Rendimiento Diario')
            plt.ylabel('Frecuencia')
            plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(crisis_dir, 'returns_distribution_crisis.png'))
    plt.close()
    
    # 3. Rendimientos acumulados durante crisis
    plt.figure(figsize=(15, 12))
    for i, (crisis_name, analysis) in enumerate(crisis_analysis.items()):
        if len(analysis['data']) > 1:
            plt.subplot(3, 1, i+1)
            returns = analysis['data']['Return'].iloc[1:]  # Eliminar el primer valor (cero)
            cumulative_returns = (1 + returns).cumprod()
            # Normalizar a 100 al inicio
            normalized_returns = cumulative_returns / cumulative_returns.iloc[0] * 100
            normalized_returns.plot(color=colors[i], linewidth=2)
            plt.axhline(y=100, color='black', linestyle='--')
            plt.title(f'Rendimientos Acumulados durante {crisis_name}')
            plt.xlabel('Fecha')
            plt.ylabel('Valor (Base 100)')
            plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(crisis_dir, 'cumulative_returns_crisis.png'))
    plt.close()
    
    # 4. Volatilidad rodante durante crisis
    plt.figure(figsize=(15, 12))
    for i, (crisis_name, analysis) in enumerate(crisis_analysis.items()):
        if len(analysis['data']) > 20:  # Necesitamos suficientes datos para la ventana
            plt.subplot(3, 1, i+1)
            returns = analysis['data']['Return'].iloc[1:]  # Eliminar el primer valor (cero)
            rolling_vol = returns.rolling(window=20).std() * np.sqrt(252) * 100  # Anualizada y en porcentaje
            rolling_vol.plot(color=colors[i], linewidth=2)
            plt.title(f'Volatilidad Rodante (20 días) durante {crisis_name}')
            plt.xlabel('Fecha')
            plt.ylabel('Volatilidad Anualizada (%)')
            plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(crisis_dir, 'rolling_volatility.png'))
    plt.close()


def save_crisis_metrics(crisis_analysis: Dict[str, Dict[str, Any]], full_period_metrics: Dict[str, float], 
                        results_dir: str) -> None:
    """
    Guarda las métricas de cada periodo de crisis y comparativas.
    
    Args:
        crisis_analysis: Resultados del análisis de crisis
        full_period_metrics: Métricas del periodo completo
        results_dir: Directorio donde guardar los resultados
    """
    # Directorio para análisis de crisis
    crisis_dir = os.path.join(results_dir, 'crisis_analysis')
    os.makedirs(crisis_dir, exist_ok=True)
    
    # Crear DataFrame comparativo de métricas
    all_metrics = {'Periodo Completo': full_period_metrics}
    for crisis_name, analysis in crisis_analysis.items():
        all_metrics[crisis_name] = analysis['metrics']
    
    # Verificar qué crisis tienen datos
    available_crisis = list(crisis_analysis.keys())
    print(f"Periodos de crisis con datos disponibles: {available_crisis}")
    
    # Crear DataFrame solo con las crisis disponibles
    metrics_comparison = pd.DataFrame(all_metrics).T
    
    # Seleccionar solo las columnas que queremos
    metrics_columns = ['total_return', 'annual_return', 'volatility', 
                      'sharpe_ratio', 'sortino_ratio', 'max_drawdown', 
                      'var_95', 'cvar_95']
    
    # Asegurarse de que todas las columnas existen
    available_columns = [col for col in metrics_columns if col in metrics_comparison.columns]
    metrics_comparison = metrics_comparison[available_columns]
    
    # Formatear para mejor visualización
    format_dict = {
        'total_return': '{:.2f}%',
        'annual_return': '{:.2f}%',
        'volatility': '{:.2f}%',
        'sharpe_ratio': '{:.2f}',
        'sortino_ratio': '{:.2f}',
        'max_drawdown': '{:.2f}%',
        'var_95': '{:.2f}%',
        'cvar_95': '{:.2f}%'
    }
    
    formatted_metrics = metrics_comparison.copy()
    for col, format_str in format_dict.items():
        if col in formatted_metrics.columns:
            formatted_metrics[col] = formatted_metrics[col].map(lambda x: format_str.format(x))
    
    # Guardar la tabla comparativa
    metrics_comparison.to_csv(os.path.join(crisis_dir, 'metrics_comparison_raw.csv'))
    formatted_metrics.to_csv(os.path.join(crisis_dir, 'metrics_comparison_formatted.csv'))
    
    # Crear un informe resumen en formato de texto
    with open(os.path.join(crisis_dir, 'crisis_analysis_summary.txt'), 'w') as f:
        f.write("ANÁLISIS DE CARTERA DQN DURANTE PERIODOS DE CRISIS\n")
        f.write("================================================\n\n")
        
        f.write("MÉTRICAS COMPARATIVAS\n")
        f.write("--------------------\n")
        f.write(formatted_metrics.to_string())
        f.write("\n\n")
        
        for crisis_name, analysis in crisis_analysis.items():
            f.write(f"ANÁLISIS DETALLADO: {crisis_name}\n")
            f.write("-" * (len(crisis_name) + 19) + "\n")
            
            metrics = analysis['metrics']
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    suffix = "%" if metric_name in ['total_return', 'annual_return', 'volatility', 'max_drawdown', 'var_95', 'cvar_95'] else ""
                    f.write(f"{metric_name}: {value:.2f}{suffix}\n")
            
            # Días extremos
            if 'worst_days' in analysis and analysis['worst_days']:
                f.write("\nDías con mayores pérdidas:\n")
                for date, ret in analysis['worst_days']:
                    f.write(f"  - {date.strftime('%Y-%m-%d')}: {ret*100:.2f}%\n")
            
            if 'best_days' in analysis and analysis['best_days']:
                f.write("\nDías con mayores ganancias:\n")
                for date, ret in analysis['best_days']:
                    f.write(f"  - {date.strftime('%Y-%m-%d')}: {ret*100:.2f}%\n")
            
            f.write("\n")
        
        # Comparación con periodo completo
        f.write("\nCOMPARACIÓN CON PERIODO COMPLETO\n")
        f.write("-------------------------------\n")
        for crisis_name, analysis in crisis_analysis.items():
            metrics = analysis['metrics']
            f.write(f"{crisis_name}:\n")
            
            # Comparar rendimiento con periodo completo de forma segura
            if 'annual_return' in metrics and 'annual_return' in full_period_metrics:
                ret_diff = metrics['annual_return'] - full_period_metrics['annual_return']
                f.write(f"  - Diferencia en rendimiento anualizado: {ret_diff:.2f}%\n")
            
            # Comparar volatilidad de forma segura
            if 'volatility' in metrics and 'volatility' in full_period_metrics:
                vol_diff = metrics['volatility'] - full_period_metrics['volatility']
                f.write(f"  - Diferencia en volatilidad anualizada: {vol_diff:.2f}%\n")
            
            # Comparar Sharpe de forma segura
            if 'sharpe_ratio' in metrics and 'sharpe_ratio' in full_period_metrics:
                sharpe_diff = metrics['sharpe_ratio'] - full_period_metrics['sharpe_ratio']
                f.write(f"  - Diferencia en Ratio Sharpe: {sharpe_diff:.2f}\n")
            
            f.write("\n")
        
        f.write("\nCONCLUSIONES DEL ANÁLISIS\n")
        f.write("------------------------\n")
        
        # Verificar si hay suficientes datos para hacer comparaciones
        if len(crisis_analysis) > 0:
            # Encuentra el periodo con mejor y peor Sharpe de forma segura
            if 'sharpe_ratio' in metrics_comparison.columns:
                best_sharpe = metrics_comparison['sharpe_ratio'].idxmax()
                worst_sharpe = metrics_comparison['sharpe_ratio'].idxmin()
                
                f.write(f"1. El mejor rendimiento ajustado al riesgo (Ratio Sharpe) se observó durante {best_sharpe}.\n")
                f.write(f"2. El peor rendimiento ajustado al riesgo se observó durante {worst_sharpe}.\n")
            
            # Evalúa si el agente DQN se desempeñó mejor que el periodo completo en alguna crisis
            better_than_full = []
            if 'sharpe_ratio' in metrics_comparison.columns and 'Periodo Completo' in metrics_comparison.index:
                full_sharpe = metrics_comparison.loc['Periodo Completo', 'sharpe_ratio']
                for crisis, row in metrics_comparison.iterrows():
                    if crisis != 'Periodo Completo' and 'sharpe_ratio' in row and row['sharpe_ratio'] > full_sharpe:
                        better_than_full.append(crisis)
            
            if better_than_full:
                f.write(f"3. El agente DQN se desempeñó mejor que el periodo completo durante: {', '.join(better_than_full)}.\n")
            else:
                f.write("3. El agente DQN no superó el rendimiento del periodo completo durante ninguna crisis.\n")
            
            # Análisis de volatilidad - versión segura sin referencias directas a crisis específicas
            f.write("\n4. Análisis de volatilidad: ")
            if 'volatility' in metrics_comparison.columns and len(crisis_analysis) > 0:
                # Encontrar la crisis con mayor volatilidad
                crisis_vol = [(name, metrics_comparison.loc[name, 'volatility']) 
                             for name in crisis_analysis.keys() 
                             if name in metrics_comparison.index]
                if crisis_vol:
                    max_vol_crisis = max(crisis_vol, key=lambda x: x[1])
                    f.write(f"El agente DQN experimentó mayor volatilidad durante {max_vol_crisis[0]} ({max_vol_crisis[1]:.2f}%).\n")
                else:
                    f.write("No hay datos suficientes para determinar el periodo con mayor volatilidad.\n")
            else:
                f.write("No hay datos suficientes para analizar la volatilidad.\n")
            
            # Análisis de drawdown - versión segura
            f.write("\n5. Análisis de drawdown: ")
            if 'max_drawdown' in metrics_comparison.columns:
                max_dd_idx = metrics_comparison['max_drawdown'].idxmax()
                max_dd_val = metrics_comparison.loc[max_dd_idx, 'max_drawdown']
                f.write(f"El periodo con mayor drawdown fue {max_dd_idx} ({max_dd_val:.2f}%).\n")
            else:
                f.write("No hay datos suficientes para analizar el drawdown.\n")
        else:
            f.write("No hay suficientes datos de periodos de crisis para realizar un análisis comparativo completo.\n")


def main() -> None:
    """Función principal del evaluador de DQN con análisis de crisis"""
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
        results_dir = os.path.join("results", f"dqn_crisis_analysis_{timestamp}")
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
        print("Evaluando agente DQN en periodo completo...")
        start_time = datetime.now()
        eval_results = evaluate_agent(agent, env, dates, render=False)
        eval_time = (datetime.now() - start_time).total_seconds()
        print(f"Evaluación completada en {eval_time:.2f} segundos ({eval_results['steps']} pasos)")
        
        # Extraer resultados
        portfolio_df = eval_results['portfolio_df']
        weights_df = eval_results['weights_df']
        full_period_metrics = eval_results['metrics']
        
        # Mostrar resumen de resultados
        print("\n==== RESUMEN DE RESULTADOS PERIODO COMPLETO ====")
        print(f"Balance final: ${eval_results['final_balance']:,.2f}")
        print(f"Rendimiento anualizado: {full_period_metrics['annual_return']:.2f}%")
        print(f"Volatilidad anualizada: {full_period_metrics['volatility']:.2f}%")
        print(f"Ratio de Sharpe: {full_period_metrics['sharpe_ratio']:.2f}")
        print(f"Ratio de Sortino: {full_period_metrics['sortino_ratio']:.2f}")
        print(f"VaR95 diario: {full_period_metrics['var_95']:.2f}%")
        print(f"Máximo Drawdown: {full_period_metrics['max_drawdown']:.2f}%")
        
        # Analizar periodos de crisis
        print("\nAnalizando periodos de crisis...")
        crisis_analysis = analyze_crisis_periods(portfolio_df, weights_df, asset_names)
        
        # Generar gráficos de todo el periodo con destacado de crisis
        plot_portfolio_value(
            portfolio_df,
            title='Evolución del Valor de la Cartera DQN con Periodos de Crisis',
            highlight_crisis=True,
            save_path=os.path.join(results_dir, 'portfolio_value_with_crisis.png')
        )
        
        # Generar gráficos específicos por crisis
        plot_crisis_analysis(crisis_analysis, results_dir)
        
        # Guardar métricas comparativas
        save_crisis_metrics(crisis_analysis, full_period_metrics, results_dir)
        
        # Guardar datos completos
        portfolio_df.to_csv(os.path.join(results_dir, 'portfolio_values_full.csv'))
        weights_df.to_csv(os.path.join(results_dir, 'weights_history_full.csv'))
        
        # Generar gráficos específicos para cada crisis individual
        for crisis_name, analysis in crisis_analysis.items():
            crisis_data = analysis['data']
            if not crisis_data.empty:
                # Gráfico de valor de cartera durante crisis específica
                plt.figure(figsize=(10, 6))
                plt.plot(crisis_data.index, crisis_data['Value'], linewidth=2)
                plt.title(f'Valor de la Cartera durante {crisis_name}')
                plt.xlabel('Fecha')
                plt.ylabel('Valor ($)')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                safe_name = crisis_name.replace(' ', '_').lower()
                plt.savefig(os.path.join(results_dir, 'crisis_analysis', f'portfolio_value_{safe_name}.png'))
                plt.close()
                
                # Análisis de retornos extremos
                if not crisis_data['Return'].empty and len(crisis_data) > 1:
                    plt.figure(figsize=(12, 5))
                    returns = crisis_data['Return'].iloc[1:]  # Eliminar el primer valor (cero)
                    plt.bar(range(len(returns)), returns * 100, color=['red' if r < 0 else 'green' for r in returns])
                    plt.title(f'Retornos Diarios durante {crisis_name}')
                    plt.xlabel('Días de Trading')
                    plt.ylabel('Rendimiento (%)')
                    plt.grid(True, axis='y', alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(results_dir, 'crisis_analysis', f'daily_returns_{safe_name}.png'))
                    plt.close()
                
                # Gráfico de drawdown
                if not analysis['drawdowns'].empty:
                    plt.figure(figsize=(10, 6))
                    analysis['drawdowns'].plot(color='red', linewidth=2)
                    plt.title(f'Drawdown durante {crisis_name}')
                    plt.xlabel('Fecha')
                    plt.ylabel('Drawdown')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(results_dir, 'crisis_analysis', f'drawdown_{safe_name}.png'))
                    plt.close()
                
                # Guardar datos específicos de la crisis
                crisis_data.to_csv(os.path.join(results_dir, 'crisis_analysis', f'data_{safe_name}.csv'))
        
        # Análisis comparativo adicional: valor acumulado normalizado para cada periodo
        plt.figure(figsize=(12, 8))
        
        # Normalizar valores al inicio de cada periodo
        normalized_series = []
        labels = []
        
        # Periodo completo (para referencia)
        full_values = portfolio_df['Value']
        normalized_full = full_values / full_values.iloc[0] * 100
        shifted_full = pd.Series(
            normalized_full.values, 
            index=[(d - normalized_full.index[0]).days for d in normalized_full.index]
        )
        normalized_series.append(shifted_full)
        labels.append('Periodo Completo')
        
        # Cada crisis
        for crisis_name, analysis in crisis_analysis.items():
            if not analysis['data'].empty:
                crisis_values = analysis['data']['Value']
                normalized_crisis = crisis_values / crisis_values.iloc[0] * 100
                # Convertir fechas a días desde el inicio para alineación
                shifted_dates = pd.Series(
                    normalized_crisis.values, 
                    index=[(d - normalized_crisis.index[0]).days for d in normalized_crisis.index]
                )
                normalized_series.append(shifted_dates)
                labels.append(crisis_name)
        
        # Determinar el máximo número de días para el eje X
        max_days = max(series.index.max() for series in normalized_series)
        
        # Graficar series normalizadas
        for i, (series, label) in enumerate(zip(normalized_series, labels)):
            plt.plot(series.index, series.values, label=label, linewidth=2)
        
        plt.title('Comparación de Rendimiento Normalizado por Periodo (Base 100)')
        plt.xlabel('Días desde el inicio del periodo')
        plt.ylabel('Valor Normalizado')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'crisis_analysis', 'normalized_comparison.png'))
        plt.close()
        
        # Crear gráfico de eficiencia: Rendimiento vs Riesgo
        plt.figure(figsize=(10, 8))
        
        # Crear gráfico de eficiencia: Rendimiento vs Riesgo
        plt.figure(figsize=(10, 8))

        # Preparar datos para el gráfico - de forma segura
        volatilities = []
        returns = []
        labels = []

        # Añadir periodo completo
        volatilities.append(full_period_metrics['volatility'])
        returns.append(full_period_metrics['annual_return'])
        labels.append('Periodo Completo')

        # Añadir periodos de crisis
        for crisis_name, analysis in crisis_analysis.items():
            volatilities.append(analysis['metrics']['volatility'])
            returns.append(analysis['metrics']['annual_return'])
            labels.append(crisis_name)

        colors = ['blue', 'red', 'orange', 'purple']

        # Graficar puntos
        for i, (volatility, return_val, label) in enumerate(zip(volatilities, returns, labels)):
            marker_size = 150 if i == 0 else 100
            plt.scatter(volatility, return_val, s=marker_size, c=colors[i % len(colors)], alpha=0.7, label=label)
            plt.annotate(label, (volatility, return_val), xytext=(10, 0), textcoords='offset points')

        # Línea de referencia Sharpe = 1
        x_range = np.linspace(0, max(volatilities) * 1.1, 100)
        plt.plot(x_range, x_range, 'r--', label='Sharpe Ratio = 1')

        plt.xlabel('Volatilidad Anualizada (%)')
        plt.ylabel('Rendimiento Anualizado (%)')
        plt.title('Análisis de Eficiencia: Rendimiento vs. Riesgo por Periodo')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'crisis_analysis', 'risk_return_analysis.png'))
        plt.close()
        
        print(f"\nAnálisis de crisis completado. Resultados guardados en: {results_dir}/crisis_analysis")
        
    except Exception as e:
        print(f"\nERROR durante la evaluación: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
   sys.exit(main())