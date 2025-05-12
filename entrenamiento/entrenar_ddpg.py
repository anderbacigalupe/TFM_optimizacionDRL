import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import json
import time
import sys

# Add the project root to Python's path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Intentamos importar tqdm, pero si no está disponible, creamos una clase sustituta
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Nota: La biblioteca 'tqdm' no está instalada. No se mostrarán barras de progreso.")
    
    # Clase sustituta simple para tqdm
    class FakeTqdm:
        def __init__(self, total, desc="", leave=True):
            self.total = total
            self.desc = desc
            self.n = 0
            self.leave = leave
        
        def update(self, n=1):
            self.n += n
            if self.n % 25 == 0 or self.n >= self.total:  # Mostrar progreso cada 25 pasos
                print(f"\r{self.desc}: {self.n}/{self.total} ({self.n*100/self.total:.1f}%)", end="")
        
        def close(self):
            if self.leave:
                print()  # Nueva línea al cerrar
    
    tqdm = FakeTqdm

# Importamos nuestro entorno y agente
from entorno.entorno_cartera import PortfolioEnv
from agentes.agente_ddpg import DDPGAgent

# Aplicamos monkey patch para asegurar que el agente acepta min_weight y tau
# Guardamos el inicializador original
original_init = DDPGAgent.__init__

# Configuración del entrenamiento
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Parámetros de entrenamiento
NUM_EPISODES = 1000  # Reducido a 1000 episodios
MAX_STEPS = 252  # Aproximadamente un año de trading
SAVE_MODEL_EVERY = 50  # Guardar el modelo cada 50 episodios (menos archivos)
EVAL_EVERY = 20  # Evaluar cada 20 episodios
EVAL_EPISODES = 5  # Número de episodios para evaluar
WARMUP_EPISODES = 10  # Aumentado a 10 episodios
PATIENCE = 50  # Para early stopping - detenerse si no hay mejora en 50 episodios

# Parámetros del agente
ACTOR_LR = 0.0007
CRITIC_LR = 0.001
GAMMA = 0.99
TAU = 0.005
BUFFER_CAPACITY = 500000  # Reducido para optimizar memoria
BATCH_SIZE = 128
HIDDEN_DIM = 64
MIN_WEIGHT = 0.05
NOISE_SIGMA = 0.1

# Parámetros de exploración
NOISE_SIGMA_START = 0.2  # Mayor ruido al inicio
NOISE_SIGMA_END = 0.05  # Menor ruido al final
NOISE_DECAY = 0.995  # Factor de decaimiento del ruido

def create_portfolio_env(data_path):
    """
    Crea el entorno de cartera a partir de datos históricos.
    """
    try:
        # Intenta cargar con la primera columna como índice (formato guardado por pandas)
        df = pd.read_csv(data_path, index_col=0)
        data = df.values
        asset_names = df.columns.tolist()
    except:
        # Si falla, intenta el formato donde la primera columna es la fecha pero no es índice
        df = pd.read_csv(data_path)
        data = df.iloc[:, 1:].values
        asset_names = df.columns[1:].tolist()
    
    print(f"Datos cargados con forma: {data.shape}")
    
    # Creamos el entorno con los datos
    env = PortfolioEnv(data=data)
    
    return env, asset_names

def evaluate_agent(agent, env, num_episodes=5, render=False):
    """
    Evalúa el rendimiento del agente DDPG en el entorno.
    
    Args:
        agent: Agente DDPG a evaluar
        env: Entorno de portafolio
        num_episodes: Número de episodios para evaluar
        render: Si se debe renderizar el entorno
        
    Returns:
        dict: Resultados de la evaluación incluyendo ratio de Sharpe
    """
    agent.eval()  # Modo evaluación (sin exploración)
    
    total_rewards = []
    final_balances = []
    all_daily_returns = []  # Acumular retornos para Sharpe
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        daily_returns = []
        previous_value = env.balance  # Inicializar valor anterior
        
        while not done:
            # Seleccionamos acción sin ruido para evaluación
            action = agent.select_action(state, add_noise=False)
            next_state, reward, done, _, info = env.step(action)
            
            if render and ep == 0:  # Solo renderizamos el primer episodio
                env.render()
            
            # Capturar retorno diario
            current_value = info.get('portfolio_value', env.balance)
            daily_return = current_value / previous_value - 1
            daily_returns.append(daily_return)
            previous_value = current_value
            
            episode_reward += reward
            state = next_state
        
        total_rewards.append(episode_reward)
        final_balances.append(env.balance)
        all_daily_returns.extend(daily_returns)
    
    # Calcular Sharpe ratio
    if len(all_daily_returns) > 0:
        avg_return = np.mean(all_daily_returns) * 252  # Anualizado
        std_return = np.std(all_daily_returns) * np.sqrt(252)  # Anualizado
        sharpe_ratio = avg_return / std_return if std_return > 0 else 0
        
        # Calcular Sortino
        neg_returns = np.array([r for r in all_daily_returns if r < 0])
        downside_dev = np.std(neg_returns) * np.sqrt(252) if len(neg_returns) > 0 else 1e-6
        sortino_ratio = avg_return / downside_dev if downside_dev > 0 else 0
        
        # Calcular máximo drawdown
        cum_returns = np.cumprod(1 + np.array(all_daily_returns))
        peak = np.maximum.accumulate(cum_returns)
        drawdown = (peak - cum_returns) / peak
        max_drawdown = np.max(drawdown) * 100 if len(drawdown) > 0 else 0
    else:
        sharpe_ratio = sortino_ratio = 0
        max_drawdown = 0
    
    agent.train()  # Volvemos a modo entrenamiento
    
    return {
        'avg_reward': np.mean(total_rewards),
        'avg_balance': np.mean(final_balances),
        'final_balances': final_balances,
        'avg_sharpe': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown
    }

def plot_training_results(rewards, balances, critic_losses, actor_losses, sharpes=None, model_dir=None):
    """
    Grafica los resultados del entrenamiento.
    
    Args:
        rewards: Lista de recompensas por episodio
        balances: Lista de balances finales por episodio
        critic_losses: Lista de pérdidas del crítico
        actor_losses: Lista de pérdidas del actor
        sharpes: Lista de ratios de Sharpe (opcional)
        model_dir: Directorio donde guardar el gráfico
    """
    num_plots = 3 if sharpes is None else 4
    plt.figure(figsize=(15, 12))
    
    # Gráfico de recompensas
    plt.subplot(num_plots, 1, 1)
    plt.plot(rewards, label='Recompensa por episodio', color='#1f77b4')
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa acumulada')
    plt.title('Rendimiento de entrenamiento')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gráfico de balance final
    plt.subplot(num_plots, 1, 2)
    plt.plot(balances, label='Balance final', color='#2ca02c')
    plt.axhline(y=1000000, color='r', linestyle='--', label='Balance inicial')
    plt.xlabel('Episodio')
    plt.ylabel('Balance ($)')
    plt.title('Balance final por episodio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gráfico de pérdidas
    plt.subplot(num_plots, 1, 3)
    if critic_losses:
        plt.plot(critic_losses, label='Pérdida del crítico', color='#d62728')
    if actor_losses:
        plt.plot(actor_losses, label='Pérdida del actor', color='#9467bd')
    plt.xlabel('Episodio')
    plt.ylabel('Valor de pérdida')
    plt.title('Evolución de las pérdidas')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gráfico de Sharpe si está disponible
    if sharpes:
        plt.subplot(num_plots, 1, 4)
        plt.plot(sharpes, label='Ratio Sharpe', color='#ff7f0e', marker='o')
        plt.xlabel('Evaluación')
        plt.ylabel('Ratio de Sharpe')
        plt.title('Evolución del Ratio de Sharpe')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if model_dir:
        plt.savefig(os.path.join(model_dir, 'training_results.png'), dpi=300)
    plt.close()
    
    # Gráfico adicional de Sharpe
    if sharpes:
        plt.figure(figsize=(10, 6))
        plt.plot(sharpes, color='#ff7f0e', marker='o')
        plt.axhline(y=max(sharpes), color='r', linestyle='--', 
                   label=f'Mejor Sharpe: {max(sharpes):.4f}')
        plt.xlabel('Evaluación')
        plt.ylabel('Ratio de Sharpe')
        plt.title('Evolución del Ratio de Sharpe')
        plt.legend()
        plt.grid(True, alpha=0.3)
        if model_dir:
            plt.savefig(os.path.join(model_dir, 'sharpe_evolution.png'), dpi=300)
        plt.close()

def save_training_metrics(metrics, model_dir):
    """
    Guarda las métricas de entrenamiento en formato JSON.
    """
    metrics_path = os.path.join(model_dir, 'training_metrics.json')
    
    # Convertimos arrays numpy a listas para serialización JSON
    serializable_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            serializable_metrics[key] = value.tolist()
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
            serializable_metrics[key] = [v.tolist() for v in value]
        elif isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            serializable_metrics[key] = str(value)  # Convertir NaN e Inf a strings
        else:
            serializable_metrics[key] = value
    
    with open(metrics_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=4)
    
    print(f"Métricas de entrenamiento guardadas en {metrics_path}")

def main():
    # Definimos la ruta a los datos
    data_path = 'data/processed/processed_prices.csv'
    
    # Verificamos que existe el archivo
    if not os.path.exists(data_path):
        print(f"Error: No se encontró el archivo {data_path}")
        print("Verificando otras rutas posibles...")
        
        # Intentamos buscar el archivo en otras ubicaciones comunes
        alternate_paths = [
            'datos/processed_prices.csv',
            'datos/precios_historicos.csv',
            './processed_prices.csv'
        ]
        
        for alt_path in alternate_paths:
            if os.path.exists(alt_path):
                data_path = alt_path
                print(f"Encontrado archivo en: {data_path}")
                break
        else:
            print("No se encontró ningún archivo de datos. Por favor, verifica la ruta.")
            return
    
    print(f"Usando datos de: {data_path}")
    
    # Creamos el entorno
    env, asset_names = create_portfolio_env(data_path)
    print(f"Entorno creado con {len(asset_names)} activos: {asset_names}")
    
    # Definimos dimensiones de estados y acciones
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"Dimensión del estado: {state_dim}")
    print(f"Dimensión de la acción: {action_dim}")
    
    # Creamos el agente DDPG
    agent = DDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=HIDDEN_DIM,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR,
        gamma=GAMMA,
        tau=TAU,
        buffer_capacity=BUFFER_CAPACITY,
        batch_size=BATCH_SIZE,
        min_weight=MIN_WEIGHT,
        noise_sigma=NOISE_SIGMA
    )
    
    # Creamos un directorio para guardar los modelos
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"modelos/ddpg_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Guardamos la configuración del entrenamiento
    config = {
        'data_path': data_path,
        'num_episodes': NUM_EPISODES,
        'max_steps': MAX_STEPS,
        'actor_lr': ACTOR_LR,
        'critic_lr': CRITIC_LR,
        'gamma': GAMMA,
        'tau': TAU,
        'buffer_capacity': BUFFER_CAPACITY,
        'batch_size': BATCH_SIZE,
        'hidden_dim': HIDDEN_DIM,
        'min_weight': MIN_WEIGHT,
        'noise_sigma': NOISE_SIGMA,
        'noise_sigma_start': NOISE_SIGMA_START,
        'noise_sigma_end': NOISE_SIGMA_END,
        'noise_decay': NOISE_DECAY,
        'assets': asset_names,
        'timestamp': timestamp
    }
    
    with open(os.path.join(model_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Listas para almacenar resultados
    episode_rewards = []
    episode_balances = []
    episode_critic_losses = []
    episode_actor_losses = []
    eval_sharpes = []  # Lista para seguimiento de Sharpe
    best_eval_balance = 0
    best_eval_sharpe = float('-inf')  # Inicializar con valor mínimo
    no_improvement_count = 0  # Contador para early stopping
    
    print("\n" + "="*50)
    print("Iniciando entrenamiento del agente DDPG")
    print("="*50 + "\n")
    
    # Fase de calentamiento (warmup)
    print(f"Fase de calentamiento: {WARMUP_EPISODES} episodios...")
    
    for episode in range(WARMUP_EPISODES):
        state, _ = env.reset()
        done = False
        step = 0
        
        while not done and step < MAX_STEPS:
            # Seleccionamos una acción (con ruido para exploración)
            action = agent.select_action(state, add_noise=True)
            
            # Ejecutamos la acción en el entorno
            next_state, reward, done, _, info = env.step(action)
            
            # Almacenamos la experiencia en el buffer, pero no actualizamos la política
            agent.memory.push(state, action, reward, next_state, done)
            
            # Actualizamos el estado
            state = next_state
            step += 1
        
        print(f"Episodio de calentamiento {episode+1}/{WARMUP_EPISODES} completado - Buffer: {len(agent.memory)}/{BUFFER_CAPACITY}")
    
    print("\n" + "="*50)
    print(f"Iniciando entrenamiento principal: {NUM_EPISODES} episodios")
    print("="*50 + "\n")
    
    start_time = time.time()
    current_noise = NOISE_SIGMA_START
    
    # Entrenamiento principal
    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        step = 0
        critic_losses = []
        actor_losses = []
        
        # Aplicar decaimiento al ruido
        current_noise = max(NOISE_SIGMA_END, current_noise * NOISE_DECAY)
        
        # Actualizar sigma de ruido en agente si es posible
        if hasattr(agent, 'noise') and hasattr(agent.noise, 'sigma'):
            agent.noise.sigma = current_noise
        
        try:
            from tqdm import tqdm
            progress_bar = tqdm(total=MAX_STEPS, desc=f"Episodio {episode+1}/{NUM_EPISODES}", leave=False)
        except ImportError:
            progress_bar = None
        
        while not done and step < MAX_STEPS:
            # Seleccionamos una acción (con ruido para exploración)
            action = agent.select_action(state, add_noise=True)
            
            # Ejecutamos la acción en el entorno
            next_state, reward, done, _, info = env.step(action)
            
            # Almacenamos la experiencia en el buffer
            agent.memory.push(state, action, reward, next_state, done)
            
            # Actualizamos el agente
            if len(agent.memory) > BATCH_SIZE:
                critic_loss, actor_loss = agent.update()
                if critic_loss is not None:
                    critic_losses.append(critic_loss)
                if actor_loss is not None:
                    actor_losses.append(actor_loss)
            
            # Actualizamos el estado y la recompensa
            state = next_state
            episode_reward += reward
            step += 1
            
            if progress_bar:
                progress_bar.update(1)
        
        if progress_bar:
            progress_bar.close()
        
        # Guardamos métricas del episodio
        episode_rewards.append(episode_reward)
        episode_balances.append(env.balance)
        
        if critic_losses:
            episode_critic_losses.append(np.mean(critic_losses))
        if actor_losses:
            episode_actor_losses.append(np.mean(actor_losses))
        
        # Tiempo transcurrido
        elapsed_time = time.time() - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Evaluamos el agente periódicamente
        if episode % EVAL_EVERY == 0 or episode == NUM_EPISODES - 1:
            print(f"\nEvaluando agente en episodio {episode+1}...")
            eval_results = evaluate_agent(agent, env, EVAL_EPISODES)
            
            # Guardar valor de Sharpe
            eval_sharpes.append(eval_results['avg_sharpe'])
            
            print(f"Episodio {episode+1}/{NUM_EPISODES} | "
                f"Tiempo: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d} | "
                f"Recompensa: {episode_reward:.4f} | "
                f"Balance: ${env.balance:.2f} | "
                f"Crítico Loss: {np.mean(critic_losses) if critic_losses else 'N/A':.6f} | "
                f"Actor Loss: {np.mean(actor_losses) if actor_losses else 'N/A':.6f} | "
                f"Eval Balance: ${eval_results['avg_balance']:.2f} | "
                f"Eval Sharpe: {eval_results['avg_sharpe']:.4f} | "
                f"Sortino: {eval_results['sortino_ratio']:.4f} | "
                f"MaxDD: {eval_results['max_drawdown']:.2f}%")
            
            # Guardamos el mejor modelo según el ratio Sharpe
            if eval_results['avg_sharpe'] > best_eval_sharpe:
                best_eval_sharpe = eval_results['avg_sharpe']
                agent.save(os.path.join(model_dir, 'best_model.pth'))
                print(f"Nuevo mejor modelo guardado con Sharpe: {best_eval_sharpe:.4f} (Balance: ${eval_results['avg_balance']:.2f})")
                no_improvement_count = 0  # Resetear contador para early stopping
            else:
                no_improvement_count += 1
                print(f"Sin mejora en Sharpe durante {no_improvement_count} evaluaciones consecutivas")
                
            # Early stopping
            if no_improvement_count >= PATIENCE:
                print(f"Early stopping después de {no_improvement_count} evaluaciones sin mejora en Sharpe.")
                break
                
        else:
            print(f"Episodio {episode+1}/{NUM_EPISODES} | "
                  f"Tiempo: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d} | "
                  f"Recompensa: {episode_reward:.4f} | "
                  f"Balance: ${env.balance:.2f} | "
                  f"Crítico Loss: {np.mean(critic_losses) if critic_losses else 'N/A':.6f} | "
                  f"Actor Loss: {np.mean(actor_losses) if actor_losses else 'N/A':.6f} | "
                  f"Ruido: {current_noise:.4f}")
        
        # Guardamos el modelo periódicamente
        if (episode + 1) % SAVE_MODEL_EVERY == 0:
            agent.save(os.path.join(model_dir, f'model_ep{episode+1}.pth'))
    
    # Guardamos el modelo final
    agent.save(os.path.join(model_dir, 'final_model.pth'))
    
    # Guardamos las métricas de entrenamiento
    training_metrics = {
        'rewards': episode_rewards,
        'balances': episode_balances,
        'critic_losses': episode_critic_losses,
        'actor_losses': episode_actor_losses,
        'sharpes': eval_sharpes,
        'best_sharpe': best_eval_sharpe,
        'training_duration': time.time() - start_time,
        'episodes_completed': episode + 1,
        'stopped_early': no_improvement_count >= PATIENCE
    }
    
    save_training_metrics(training_metrics, model_dir)
    
    # Graficamos los resultados
    plot_training_results(
        episode_rewards, 
        episode_balances, 
        episode_critic_losses, 
        episode_actor_losses,
        eval_sharpes,
        model_dir
    )
    
    # Evaluación final más detallada
    print("\n" + "="*50)
    print("Evaluación final del modelo")
    print("="*50 + "\n")
    
    final_eval = evaluate_agent(agent, env, EVAL_EPISODES * 2, render=True)
    
    print("\nResultados de la evaluación final:")
    print(f"Recompensa promedio: {final_eval['avg_reward']:.4f}")
    print(f"Balance promedio: ${final_eval['avg_balance']:.2f}")
    print(f"Ratio Sharpe: {final_eval['avg_sharpe']:.4f}")
    print(f"Ratio Sortino: {final_eval['sortino_ratio']:.4f}")
    print(f"Máximo Drawdown: {final_eval['max_drawdown']:.2f}%")
    print(f"Mejor balance: ${max(final_eval['final_balances']):.2f}")
    print(f"Peor balance: ${min(final_eval['final_balances']):.2f}")
    
    # Guardamos las métricas de evaluación
    evaluation_metrics = {
        'avg_reward': final_eval['avg_reward'],
        'avg_balance': final_eval['avg_balance'],
        'avg_sharpe': final_eval['avg_sharpe'],
        'sortino_ratio': final_eval['sortino_ratio'],
        'max_drawdown': final_eval['max_drawdown'],
        'final_balances': final_eval['final_balances'],
        'best_balance': max(final_eval['final_balances']),
        'worst_balance': min(final_eval['final_balances'])
    }
    
    with open(os.path.join(model_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(evaluation_metrics, f, indent=4)
    
    print(f"\nEvaluación completada. Resultados guardados en {model_dir}")
    print(f"Mejor modelo guardado en: {os.path.join(model_dir, 'best_model.pth')}")
    
    return os.path.join(model_dir, 'best_model.pth')

if __name__ == "__main__":
    main()