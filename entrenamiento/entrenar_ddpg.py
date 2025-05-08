import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import json
import time

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

# Configuración del entrenamiento
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Parámetros de entrenamiento
NUM_EPISODES = 200
MAX_STEPS = 252  # Aproximadamente un año de trading (252 días)
SAVE_MODEL_EVERY = 20  # Guardar el modelo cada 20 episodios
EVAL_EPISODES = 5  # Número de episodios para evaluar
WARMUP_EPISODES = 5  # Episodios de calentamiento sin actualizar política

# Parámetros del agente
ACTOR_LR = 0.0007
CRITIC_LR = 0.001
GAMMA = 0.99
TAU = 0.005
BUFFER_CAPACITY = 1000000
BATCH_SIZE = 128
HIDDEN_DIM = 64
MIN_WEIGHT = 0.05
NOISE_SIGMA = 0.1

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
    agent.eval()  # Modo evaluación (sin exploración)
    
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
            action = agent.select_action(state, add_noise=False)  # Sin ruido durante evaluación
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
    
    agent.train()  # Volvemos a modo entrenamiento
    
    return {
        'avg_reward': np.mean(total_rewards),
        'avg_balance': np.mean(final_balances),
        'final_balances': final_balances,
        'portfolio_values': portfolio_values,
        'weights_history': weights_history
    }

def plot_training_results(rewards, balances, critic_losses, actor_losses, model_dir):
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
    
    # Gráfico de pérdidas del crítico
    if critic_losses:  # Si hay datos de pérdidas
        plt.subplot(2, 2, 3)
        plt.plot(critic_losses, label='Pérdida del crítico')
        plt.xlabel('Episodio')
        plt.ylabel('Valor de pérdida')
        plt.title('Evolución de la pérdida del crítico')
        plt.legend()
        plt.grid(True)
    
    # Gráfico de pérdidas del actor
    if actor_losses:  # Si hay datos de pérdidas
        plt.subplot(2, 2, 4)
        plt.plot(actor_losses, label='Pérdida del actor')
        plt.xlabel('Episodio')
        plt.ylabel('Valor de pérdida')
        plt.title('Evolución de la pérdida del actor')
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
    best_eval_balance = 0
    
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
    
    # Entrenamiento principal
    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        step = 0
        critic_losses = []
        actor_losses = []
        
        progress_bar = tqdm(total=MAX_STEPS, desc=f"Episodio {episode+1}/{NUM_EPISODES}", leave=False)
        
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
            
            progress_bar.update(1)
        
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
        
        # Evaluamos el agente cada 10 episodios o en el último
        if episode % 10 == 0 or episode == NUM_EPISODES - 1:
            print(f"\nEvaluando agente en episodio {episode+1}...")
            eval_results = evaluate_agent(agent, env, EVAL_EPISODES)
            
            print(f"Episodio {episode+1}/{NUM_EPISODES} | "
                  f"Tiempo: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d} | "
                  f"Recompensa: {episode_reward:.4f} | "
                  f"Balance: ${env.balance:.2f} | "
                  f"Crítico Loss: {np.mean(critic_losses) if critic_losses else 'N/A':.6f} | "
                  f"Actor Loss: {np.mean(actor_losses) if actor_losses else 'N/A':.6f} | "
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
                  f"Crítico Loss: {np.mean(critic_losses) if critic_losses else 'N/A':.6f} | "
                  f"Actor Loss: {np.mean(actor_losses) if actor_losses else 'N/A':.6f}")
        
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
        'best_balance': best_eval_balance,
        'training_duration': time.time() - start_time
    }
    
    save_training_metrics(training_metrics, model_dir)
    
    # Graficamos los resultados
    plot_training_results(
        episode_rewards, 
        episode_balances, 
        episode_critic_losses, 
        episode_actor_losses, 
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