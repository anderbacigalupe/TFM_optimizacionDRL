import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import gymnasium as gym
import inspect
import time
import json
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
from agentes.agente_dqn import DQNAgent

# Aplicamos monkey patch para asegurar que el agente acepta min_weight y tau
# Guardamos el inicializador original
original_init = DQNAgent.__init__

# Verificamos si ya tiene los parámetros que necesitamos
sig = inspect.signature(original_init)
if 'min_weight' not in sig.parameters or 'tau' not in sig.parameters:
    # Definimos un nuevo inicializador que incluye los parámetros faltantes
    def new_init(self, state_dim, action_dim, n_discrete_bins=10, learning_rate=1e-3, 
                gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                buffer_capacity=10000, batch_size=64, min_weight=0.05, tau=0.005):
        
        # Guarda los nuevos parámetros
        original_init(self, state_dim, action_dim, n_discrete_bins, learning_rate,
                    gamma, epsilon_start, epsilon_end, epsilon_decay,
                    buffer_capacity, batch_size)
        
        # Añade los atributos necesarios para tu implementación actualizada
        self.min_weight = min_weight
        self.tau = tau
        
        # Configura el dispositivo si no está ya configurado
        if not hasattr(self, 'device'):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Añadido soporte para GPU. Usando dispositivo: {self.device}")
            
        # Añadir actualizaciones adicionales si el agente no tiene otras funciones
        if not hasattr(self, 'update_target_network') or 'use_soft_update' not in inspect.signature(self.update_target_network).parameters:
            def update_target_network(slf, use_soft_update=True):
                if use_soft_update and hasattr(slf, 'tau'):
                    # Soft update: τ*θ_policy + (1-τ)*θ_target
                    for target_param, policy_param in zip(slf.target_net.parameters(), slf.policy_net.parameters()):
                        target_param.data.copy_(
                            slf.tau * policy_param.data + (1.0 - slf.tau) * target_param.data
                        )
                else:
                    # Hard update: θ_target = θ_policy
                    slf.target_net.load_state_dict(slf.policy_net.state_dict())
            
            # Reemplazar o añadir el método
            DQNAgent.update_target_network = update_target_network
        
    # Reemplaza el inicializador
    DQNAgent.__init__ = new_init
    print("DQNAgent actualizado para soportar peso mínimo y tau")

# Configuración del entrenamiento
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Parámetros de entrenamiento
NUM_EPISODES = 200  # Aumentado para mejor convergencia
MAX_STEPS = 252  # Aproximadamente un año de trading (252 días)
UPDATE_TARGET_EVERY = 5  # Actualizar la red objetivo más frecuentemente
SAVE_MODEL_EVERY = 20  # Guardar el modelo cada 20 episodios
EVAL_EPISODES = 5  # Número de episodios para evaluar
WARMUP_EPISODES = 5  # Episodios de calentamiento sin actualizar política

# Parámetros del agente
LEARNING_RATE = 0.0003  # Reducido para mayor estabilidad
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.99  # Más lento para mejor exploración
BUFFER_CAPACITY = 20000  # Aumentado para más experiencia
BATCH_SIZE = 128  # Aumentado para mejor estadística
N_DISCRETE_BINS = 5  # Número de valores discretos por activo
MIN_WEIGHT = 0.05  # Peso mínimo por activo (5%)
TAU = 0.005  # Factor para soft update

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
    Evalúa el rendimiento del agente en el entorno.
    """
    total_rewards = []
    final_balances = []
    portfolio_values = []
    weights_history = []
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        ep_portfolio_values = [env.balance]
        ep_weights_history = []
        
        step = 0
        while not done:
            action = agent.select_action(state, training=False)  # No exploración
            next_state, reward, done, _, info = env.step(action)
            
            if render and ep == 0:  # Solo renderizamos el primer episodio
                env.render()
            
            episode_reward += reward
            state = next_state
            step += 1
            
            ep_portfolio_values.append(env.balance)
            ep_weights_history.append(env.portfolio_weights.copy())
        
        total_rewards.append(episode_reward)
        final_balances.append(env.balance)
        
        if len(portfolio_values) == 0 or ep == 0:
            portfolio_values = ep_portfolio_values
            weights_history = ep_weights_history
    
    return {
        'avg_reward': np.mean(total_rewards),
        'avg_balance': np.mean(final_balances),
        'final_balances': final_balances,
        'portfolio_values': portfolio_values,
        'weights_history': weights_history
    }

def plot_training_results(rewards, balances, losses, epsilons, model_dir):
    """
    Grafica los resultados del entrenamiento.
    """
    plt.figure(figsize=(15, 12))
    
    # Gráfico de recompensas
    plt.subplot(2, 2, 1)
    plt.plot(rewards, label='Recompensa por episodio')
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa acumulada')
    plt.title('Rendimiento de entrenamiento')
    plt.legend()
    plt.grid(True)
    
    # Gráfico de balance final
    plt.subplot(2, 2, 2)
    plt.plot(balances, label='Balance final')
    plt.axhline(y=1000000, color='r', linestyle='--', label='Balance inicial')
    plt.xlabel('Episodio')
    plt.ylabel('Balance ($)')
    plt.title('Balance final por episodio')
    plt.legend()
    plt.grid(True)
    
    # Gráfico de pérdidas
    if losses:  # Si hay datos de pérdidas
        plt.subplot(2, 2, 3)
        plt.plot(losses, label='Pérdida')
        plt.xlabel('Actualización')
        plt.ylabel('Valor de pérdida')
        plt.title('Evolución de la pérdida')
        plt.legend()
        plt.grid(True)
    
    # Gráfico de epsilon
    plt.subplot(2, 2, 4)
    plt.plot(epsilons, label='Epsilon (exploración)')
    plt.xlabel('Episodio')
    plt.ylabel('Valor de epsilon')
    plt.title('Decaimiento de epsilon')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'training_results.png'))
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
    
    # Creamos el agente DQN con los parámetros mejorados
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        n_discrete_bins=N_DISCRETE_BINS,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
        buffer_capacity=BUFFER_CAPACITY,
        batch_size=BATCH_SIZE
    )
    
    agent.min_weight = MIN_WEIGHT
    agent.tau = TAU
    
    # Creamos un directorio para guardar los modelos
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"modelos/dqn_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Guardamos la configuración del entrenamiento
    config = {
        'data_path': data_path,
        'num_episodes': NUM_EPISODES,
        'max_steps': MAX_STEPS,
        'learning_rate': LEARNING_RATE,
        'gamma': GAMMA,
        'epsilon_start': EPSILON_START,
        'epsilon_end': EPSILON_END,
        'epsilon_decay': EPSILON_DECAY,
        'buffer_capacity': BUFFER_CAPACITY,
        'batch_size': BATCH_SIZE,
        'n_discrete_bins': N_DISCRETE_BINS,
        'min_weight': MIN_WEIGHT,
        'tau': TAU,
        'assets': asset_names,
        'timestamp': timestamp
    }
    
    with open(os.path.join(model_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Listas para almacenar resultados
    episode_rewards = []
    episode_balances = []
    episode_losses = []
    episode_epsilons = []
    best_eval_balance = 0
    
    print("\n" + "="*50)
    print("Iniciando entrenamiento del agente DQN")
    print("="*50 + "\n")
    
    # Fase de calentamiento (warmup)
    print(f"Fase de calentamiento: {WARMUP_EPISODES} episodios...")
    
    for episode in range(WARMUP_EPISODES):
        state, _ = env.reset()
        done = False
        step = 0
        
        while not done and step < MAX_STEPS:
            # Seleccionamos una acción aleatoria
            action = agent.select_action(state, training=True)
            
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
    
    # Entrenamiento principal
    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        step = 0
        episode_loss = []
        
        progress_bar = tqdm(total=MAX_STEPS, desc=f"Episodio {episode+1}/{NUM_EPISODES}", leave=False)
        
        while not done and step < MAX_STEPS:
            # Seleccionamos una acción
            action = agent.select_action(state)
            
            # Ejecutamos la acción en el entorno
            next_state, reward, done, _, info = env.step(action)
            
            # Almacenamos la experiencia en el buffer
            agent.memory.push(state, action, reward, next_state, done)
            
            # Actualizamos el agente
            if len(agent.memory) > BATCH_SIZE:
                loss = agent.update()
                if loss is not None:
                    episode_loss.append(loss)
            
            # Actualizamos el estado y la recompensa
            state = next_state
            episode_reward += reward
            step += 1
            
            # Actualizamos la red objetivo con soft update en cada paso
            try:
                agent.update_target_network(use_soft_update=True)
            except TypeError:
                # Si el método no acepta el parámetro, usamos la versión sin parámetros
                agent.update_target_network()
            
            progress_bar.update(1)
        
        progress_bar.close()
        
        # Actualizamos epsilon
        agent.update_epsilon()
        
        # Actualizamos la red objetivo con hard update periódicamente
        if episode % UPDATE_TARGET_EVERY == 0:
            try:
                agent.update_target_network(use_soft_update=False)
            except TypeError:
                # Si el método no acepta el parámetro, usamos la versión sin parámetros
                agent.update_target_network()
            print("Red objetivo actualizada (hard update)")
        
        # Guardamos métricas del episodio
        episode_rewards.append(episode_reward)
        episode_balances.append(env.balance)
        episode_epsilons.append(agent.epsilon)
        if episode_loss:
            episode_losses.append(np.mean(episode_loss))
        
        # Tiempo transcurrido
        elapsed_time = time.time() - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Evaluamos el agente cada 10 episodios o en el último
        if episode % 10 == 0 or episode == NUM_EPISODES - 1:
            print(f"\nEvaluando agente en episodio {episode+1}...")
            eval_results = evaluate_agent(agent, env, EVAL_EPISODES)
            
            print(f"Episodio {episode+1}/{NUM_EPISODES} | "
                  f"Tiempo: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d} | "
                  f"Recompensa: {episode_reward:.4f} | "
                  f"Balance: ${env.balance:.2f} | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"Pérdida: {np.mean(episode_loss) if episode_loss else 'N/A':.6f} | "
                  f"Eval Balance: ${eval_results['avg_balance']:.2f}")
            
            # Guardamos el mejor modelo según la evaluación
            if eval_results['avg_balance'] > best_eval_balance:
                best_eval_balance = eval_results['avg_balance']
                agent.save(os.path.join(model_dir, 'best_model.pth'))
                print(f"Nuevo mejor modelo guardado con balance: ${best_eval_balance:.2f}")
        else:
            print(f"Episodio {episode+1}/{NUM_EPISODES} | "
                  f"Tiempo: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d} | "
                  f"Recompensa: {episode_reward:.4f} | "
                  f"Balance: ${env.balance:.2f} | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"Pérdida: {np.mean(episode_loss) if episode_loss else 'N/A':.6f}")
        
        # Guardamos el modelo periódicamente
        if (episode + 1) % SAVE_MODEL_EVERY == 0:
            agent.save(os.path.join(model_dir, f'model_ep{episode+1}.pth'))
    
    # Guardamos el modelo final
    agent.save(os.path.join(model_dir, 'final_model.pth'))
    
    # Guardamos las métricas de entrenamiento
    training_metrics = {
        'rewards': episode_rewards,
        'balances': episode_balances,
        'losses': episode_losses,
        'epsilons': episode_epsilons,
        'best_balance': best_eval_balance,
        'training_duration': time.time() - start_time
    }
    
    save_training_metrics(training_metrics, model_dir)
    
    # Graficamos los resultados
    plot_training_results(episode_rewards, episode_balances, episode_losses, episode_epsilons, model_dir)
    
    # Evaluación final más detallada
    print("\n" + "="*50)
    print("Evaluación final del modelo")
    print("="*50 + "\n")
    
    final_eval = evaluate_agent(agent, env, EVAL_EPISODES * 2, render=True)
    
    print("\nResultados de la evaluación final:")
    print(f"Recompensa promedio: {final_eval['avg_reward']:.4f}")
    print(f"Balance promedio: ${final_eval['avg_balance']:.2f}")
    print(f"Mejor balance: ${max(final_eval['final_balances']):.2f}")
    print(f"Peor balance: ${min(final_eval['final_balances']):.2f}")
    
    # Guardamos las métricas de evaluación
    evaluation_metrics = {
        'avg_reward': final_eval['avg_reward'],
        'avg_balance': final_eval['avg_balance'],
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