import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import torch.nn.functional as F
import json
import time
import torch.nn as nn
import gc  # For garbage collection
import psutil  # For memory monitoring
import torch.cuda  # For CUDA memory functions
import random

# Add the project root to Python's path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Intentamos importar tqdm para barras de progreso
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
            if self.n % 25 == 0 or self.n >= self.total:
                print(f"\r{self.desc}: {self.n}/{self.total} ({self.n*100/self.total:.1f}%)", end="")
        
        def close(self):
            if self.leave:
                print()
    
    tqdm = FakeTqdm

# Importamos nuestro entorno y agente
from entorno.entorno_cartera import PortfolioEnv
from agentes.agente_ddpg import DDPGAgent

# Function to monitor memory usage
def monitor_memory():
    process = psutil.Process()
    # RAM usage in MB
    ram_usage = process.memory_info().rss / (1024 * 1024)
    
    gpu_memory = "N/A"
    if torch.cuda.is_available():
        # GPU memory in MB
        gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
    
    return ram_usage, gpu_memory

# Function to clear memory
def clear_memory():
    # Force garbage collection
    gc.collect()
    
    # Clear CUDA cache if using GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # Print memory stats
    ram, gpu = monitor_memory()
    print(f"Memory after clearing - RAM: {ram:.2f} MB, GPU: {gpu}")

# More memory-efficient replay buffer
class MemoryEfficientReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.device = device
        self.buffer = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        # Store as numpy arrays rather than tensors when not needed
        self.buffer[self.position] = (
            state.copy(),  # Using copy() to ensure we don't store references
            action.copy(),
            float(reward),  # Store as scalar when possible
            next_state.copy(),
            bool(done)  # Store as boolean
        )
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        # Only convert to tensors when sampling
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        
        # Convert to tensors just before returning
        return (
            torch.FloatTensor(np.array(state)).to(self.device),
            torch.FloatTensor(np.array(action)).to(self.device),
            torch.FloatTensor(np.array(reward)).unsqueeze(1).to(self.device),
            torch.FloatTensor(np.array(next_state)).to(self.device),
            torch.FloatTensor(np.array(done)).unsqueeze(1).to(self.device)
        )
    
    def __len__(self):
        return len(self.buffer)
    
    def clear_old_experiences(self, keep_fraction=0.5):
        """Clear older experiences to free memory"""
        if len(self.buffer) > 1000:  # Only if we have enough samples
            # Sort by recency (if we know the order) or just keep the last N
            keep_count = int(len(self.buffer) * keep_fraction)
            self.buffer = self.buffer[-keep_count:]
            self.position = len(self.buffer) % self.capacity

# Modificamos la clase OUNoise para mejorar la exploración
class EnhancedOUNoise:
    """Versión mejorada de Ornstein-Uhlenbeck Noise para mejor exploración"""
    def __init__(self, action_dim, mu=0, theta=0.3, sigma_start=0.3, sigma_end=0.05, decay_rate=0.9995):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta  # Mayor theta para más reversion a la media
        self.sigma = sigma_start
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.decay_rate = decay_rate
        self.reset()
        
    def reset(self):
        # Inicialización más diversa para evitar estancamiento
        self.state = np.random.normal(self.mu, 0.2, self.action_dim)
        
    def decay_sigma(self):
        """Reduce gradualmente el valor de sigma."""
        self.sigma = max(self.sigma_end, self.sigma * self.decay_rate)
        
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        # Añadimos ocasionalmente perturbación extra para salir de mínimos locales
        if np.random.random() < 0.05:  # 5% de probabilidad
            dx += np.random.normal(0, self.sigma * 2, self.action_dim)
        self.state = x + dx
        return self.state

# Clase para early stopping
class SimpleEarlyStopping:
    def __init__(self, patience=50, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_sharpe = -float('inf')
        self.best_return = -float('inf')
        
    def __call__(self, sharpe_ratio, portfolio_return=None):
        improvement = False
        
        if sharpe_ratio > self.best_sharpe + self.min_delta:
            self.best_sharpe = sharpe_ratio
            improvement = True
        
        if portfolio_return is not None and portfolio_return > self.best_return + self.min_delta:
            self.best_return = portfolio_return
            if sharpe_ratio >= self.best_sharpe * 0.95:  # El Sharpe no debe caer demasiado
                improvement = True
        
        if improvement:
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

# Configuración del entrenamiento
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Parámetros actualizados de entrenamiento
NUM_EPISODES = 2000           # Aumentado a 2000 episodios
MAX_STEPS = 252               # Aproximadamente un año de trading
SAVE_MODEL_EVERY = 50         # Guardar el modelo cada 50 episodios
EVAL_EVERY = 20               # Evaluar cada 20 episodios
EVAL_EPISODES = 10            # Aumentado a 10 episodios para evaluación más robusta
WARMUP_EPISODES = 20          # Aumentado a 20 episodios de calentamiento
PATIENCE = 50                 # Para early stopping
MEMORY_CHECK_INTERVAL = 5     # Comprobar memoria cada 5 episodios
MEMORY_CLEANUP_THRESHOLD = 4000  # Limpiar memoria si RAM > 4GB

# Parámetros del agente actualizados
ACTOR_LR = 0.0003             # Reducido para mayor estabilidad
CRITIC_LR = 0.001
GAMMA = 0.99
TAU = 0.005
BUFFER_CAPACITY = 500000
BATCH_SIZE = 256              # Aumentado a 256
HIDDEN_DIM = 128
MIN_WEIGHT = 0.05

# Parámetros de exploración mejorados
NOISE_SIGMA_START = 0.3       # Aumentado para mayor exploración
NOISE_SIGMA_END = 0.05
NOISE_DECAY = 0.9995          # Decaimiento más lento

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """Calcula el ratio de Sharpe a partir de los retornos diarios."""
    if len(returns) == 0:
        return 0.0
        
    # Asumimos retornos diarios y anualización usando sqrt(252)
    mean_return = np.mean(returns) * 252  # Anualizado
    volatility = np.std(returns) * np.sqrt(252)  # Anualizada
    
    if volatility == 0:
        return 0.0
        
    return (mean_return - risk_free_rate) / volatility

def create_portfolio_env(data_path):
    """Crea el entorno de cartera a partir de datos históricos."""
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
    """Evalúa el rendimiento del agente en el entorno."""
    agent.eval()  # Modo evaluación (sin exploración)
    
    total_rewards = []
    final_balances = []
    portfolio_values = []
    weights_history = []
    daily_returns = []
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        ep_portfolio_values = [env.balance]
        ep_weights_history = []
        ep_daily_returns = []
        
        step = 0
        prev_balance = env.balance
        
        while not done:
            action = agent.select_action(state, add_noise=False)  # Sin ruido durante evaluación
            next_state, reward, done, _, info = env.step(action)
            
            if render and ep == 0:  # Solo renderizamos el primer episodio
                env.render()
            
            # Calcular retorno diario
            daily_return = (env.balance / prev_balance) - 1
            ep_daily_returns.append(daily_return)
            prev_balance = env.balance
            
            episode_reward += reward
            state = next_state
            step += 1
            
            ep_portfolio_values.append(env.balance)
            ep_weights_history.append(env.portfolio_weights.copy())
        
        total_rewards.append(episode_reward)
        final_balances.append(env.balance)
        daily_returns.extend(ep_daily_returns)
        
        if len(portfolio_values) == 0 or ep == 0:
            portfolio_values = ep_portfolio_values
            weights_history = ep_weights_history
    
    # Calcular ratio de Sharpe
    sharpe = calculate_sharpe_ratio(daily_returns)
    
    agent.train()  # Volvemos a modo entrenamiento
    
    return {
        'avg_reward': np.mean(total_rewards),
        'avg_balance': np.mean(final_balances),
        'final_balances': final_balances,
        'portfolio_values': portfolio_values,
        'weights_history': weights_history,
        'sharpe_ratio': sharpe,
        'daily_returns': daily_returns
    }

def select_action_with_enhanced_exploration(agent, state, episode, total_episodes):
    """Selección de acción con exploración mejorada para entrenamiento"""
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
    
    with torch.no_grad():
        # Aplicar temperatura variable al softmax del actor
        temperature = max(1.0, 2.0 * (1 - episode/total_episodes))
        
        # Obtener la acción base del actor
        x = F.relu(agent.actor.fc1(state_tensor))
        x = F.relu(agent.actor.fc2(x))
        logits = agent.actor.fc3(x)
        
        # Aplicar softmax con temperatura para controlar distribución
        action = F.softmax(logits / temperature, dim=-1).cpu().numpy().squeeze(0)
    
    # Añadir ruido para exploración
    noise = agent.noise.sample()
    
    # Aplicar ruido y normalizarlo
    action = action + noise
    action = np.clip(action, 0, 1)
    
    # Añadir perturbación extra ocasional para evitar estancamiento
    if np.random.random() < 0.3:  # 30% de probabilidad
        # Seleccionar aleatoriamente 2-3 activos para perturbar
        n_assets = action.shape[0]
        n_perturb = np.random.randint(2, 4)
        perturb_indices = np.random.choice(n_assets, n_perturb, replace=False)
        
        # Aplicar perturbación significativa a los activos seleccionados
        perturbation = np.zeros_like(action)
        perturbation[perturb_indices] = np.random.uniform(0.2, 0.5, size=n_perturb)
        
        # Asegurar que no se viole la restricción de suma=1
        adjustment = np.sum(perturbation) / (n_assets - n_perturb)
        for i in range(n_assets):
            if i not in perturb_indices:
                action[i] = max(0, action[i] - adjustment)
        
        # Aplicar la perturbación
        action[perturb_indices] += perturbation[perturb_indices]
    
    # Normalizar para que los pesos sumen 1
    if np.sum(action) > 0:
        action = action / np.sum(action)
    
    return action

def diversified_sample_from_buffer(buffer, batch_size, device):
    """Muestreo diversificado del buffer de experiencia"""
    if len(buffer) < batch_size:
        return None
    
    # 80% muestreo aleatorio regular
    regular_size = int(batch_size * 0.8)
    regular_indices = np.random.choice(len(buffer), regular_size, replace=False)
    
    # 20% muestreo centrado en experiencias con recompensas más altas
    remaining_indices = [i for i in range(len(buffer)) if i not in regular_indices]
    
    # Obtener recompensas de las experiencias restantes
    remaining_rewards = [buffer[i][2] for i in remaining_indices]
    
    # Calcular probabilidades proporcionales a las recompensas
    # (añadimos un offset para manejar recompensas negativas)
    reward_offset = abs(min(remaining_rewards)) + 1e-3 if min(remaining_rewards) < 0 else 0
    adjusted_rewards = [r + reward_offset for r in remaining_rewards]
    
    # Normalizar para obtener probabilidades
    total_reward = sum(adjusted_rewards)
    if total_reward > 0:
        probs = [r / total_reward for r in adjusted_rewards]
        diverse_indices = np.random.choice(
            remaining_indices, 
            size=batch_size - regular_size,
            replace=False if len(remaining_indices) >= (batch_size - regular_size) else True,
            p=probs
        )
    else:
        diverse_indices = np.random.choice(
            remaining_indices,
            size=batch_size - regular_size,
            replace=False if len(remaining_indices) >= (batch_size - regular_size) else True
        )
    
    # Combinar índices
    all_indices = np.concatenate([regular_indices, diverse_indices])
    
    # Obtener las experiencias
    batch = [buffer[i] for i in all_indices]
    state, action, reward, next_state, done = zip(*batch)
    
    # Convertir a tensores y mover a GPU
    return (
        torch.FloatTensor(np.array(state)).to(device),
        torch.FloatTensor(np.array(action)).to(device),
        torch.FloatTensor(np.array(reward)).unsqueeze(1).to(device),
        torch.FloatTensor(np.array(next_state)).to(device),
        torch.FloatTensor(np.array(done)).unsqueeze(1).to(device)
    )

def plot_training_results(rewards, balances, critic_losses, actor_losses, sharpe_ratios, model_dir):
    """Grafica los resultados del entrenamiento."""
    plt.figure(figsize=(15, 12))
    
    # Gráfico de recompensas
    plt.subplot(2, 3, 1)
    plt.plot(rewards, label='Recompensa por episodio')
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa acumulada')
    plt.title('Rendimiento de entrenamiento')
    plt.legend()
    plt.grid(True)
    
    # Gráfico de balance final
    plt.subplot(2, 3, 2)
    plt.plot(balances, label='Balance final')
    plt.axhline(y=1000000, color='r', linestyle='--', label='Balance inicial')
    plt.xlabel('Episodio')
    plt.ylabel('Balance ($)')
    plt.title('Balance final por episodio')
    plt.legend()
    plt.grid(True)
    
    # Gráfico de pérdidas del crítico
    if critic_losses:  # Si hay datos de pérdidas
        plt.subplot(2, 3, 3)
        plt.plot(critic_losses, label='Pérdida del crítico')
        plt.xlabel('Episodio')
        plt.ylabel('Valor de pérdida')
        plt.title('Evolución de la pérdida del crítico')
        plt.legend()
        plt.grid(True)
    
    # Gráfico de pérdidas del actor
    if actor_losses:  # Si hay datos de pérdidas
        plt.subplot(2, 3, 4)
        plt.plot(actor_losses, label='Pérdida del actor')
        plt.xlabel('Episodio')
        plt.ylabel('Valor de pérdida')
        plt.title('Evolución de la pérdida del actor')
        plt.legend()
        plt.grid(True)
    
    # Gráfico de Sharpe ratio
    if sharpe_ratios:
        plt.subplot(2, 3, 5)
        plt.plot(sharpe_ratios, label='Ratio Sharpe')
        plt.xlabel('Evaluación')
        plt.ylabel('Ratio Sharpe')
        plt.title('Evolución del Ratio Sharpe')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'training_results.png'))
    plt.close()

def save_training_metrics(metrics, model_dir):
    """Guarda las métricas de entrenamiento en formato JSON."""
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
    
    # Initial memory check
    print("Initial memory usage:")
    ram, gpu = monitor_memory()
    print(f"RAM: {ram:.2f} MB, GPU: {gpu}")
    
    # Variables para el seguimiento de velocidad
    training_speeds = []
    last_episode_time = time.time()
    
    # Factor de actualización (inicialmente actualiza en cada paso)
    UPDATE_FREQUENCY = 1
    
    # Creamos el agente DDPG con los parámetros mejorados
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
        noise_sigma=NOISE_SIGMA_START
    )
    
    # Reemplazamos el buffer original con nuestra versión optimizada para memoria
    original_buffer = agent.memory
    agent.memory = MemoryEfficientReplayBuffer(BUFFER_CAPACITY, agent.device)
    # Transferir experiencias existentes si hay alguna
    if len(original_buffer) > 0:
        for experience in original_buffer.buffer:
            if experience is not None:
                agent.memory.push(*experience)
    
    # Liberamos memoria del buffer original
    del original_buffer
    clear_memory()
    
    # Reemplazamos el ruido estándar por nuestra versión mejorada
    agent.noise = EnhancedOUNoise(
        action_dim=action_dim,
        mu=0,
        theta=0.3,  # Mayor theta para más reversion a la media
        sigma_start=NOISE_SIGMA_START,
        sigma_end=NOISE_SIGMA_END,
        decay_rate=NOISE_DECAY
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
    eval_sharpe_ratios = []
    best_eval_balance = 0
    best_eval_sharpe = 0
    
    # Inicializar early stopping
    early_stopping = SimpleEarlyStopping(patience=PATIENCE)
    
    print("\n" + "="*50)
    print("Iniciando entrenamiento del agente DDPG")
    print("="*50 + "\n")
    
    # Fase de calentamiento (warmup)
    print(f"Fase de calentamiento: {WARMUP_EPISODES} episodios...")
    
    for episode in range(WARMUP_EPISODES):
        state, _ = env.reset()
        agent.noise.reset()  # Resetear el estado del ruido
        done = False
        step = 0
        
        while not done and step < MAX_STEPS:
            # Usar exploración mejorada durante calentamiento
            action = select_action_with_enhanced_exploration(agent, state, episode, WARMUP_EPISODES)
            
            # Ejecutamos la acción en el entorno
            next_state, reward, done, _, info = env.step(action)
            
            # Almacenamos la experiencia en el buffer
            agent.memory.push(state, action, reward, next_state, done)
            
            # Actualizamos el estado
            state = next_state
            step += 1
        
        print(f"Episodio de calentamiento {episode+1}/{WARMUP_EPISODES} completado - Buffer: {len(agent.memory)}/{BUFFER_CAPACITY}")
        
        # Limpiar memoria cada 5 episodios de calentamiento
        if (episode + 1) % 5 == 0:
            clear_memory()
    
    print("\n" + "="*50)
    print(f"Iniciando entrenamiento principal: {NUM_EPISODES} episodios")
    print("="*50 + "\n")
    
    start_time = time.time()
    
    # Entrenamiento principal
    for episode in range(NUM_EPISODES):
        episode_start_time = time.time()
        state, _ = env.reset()
        agent.noise.reset()  # Resetear el ruido para cada episodio
        done = False
        episode_reward = 0
        step = 0
        critic_losses = []
        actor_losses = []
        
        progress_bar = tqdm(total=MAX_STEPS, desc=f"Episodio {episode+1}/{NUM_EPISODES}", leave=False)
        
        while not done and step < MAX_STEPS:
            # Usar la selección de acción mejorada
            action = select_action_with_enhanced_exploration(agent, state, episode, NUM_EPISODES)
            
            # Ejecutamos la acción en el entorno
            next_state, reward, done, _, info = env.step(action)
            
            # Almacenamos la experiencia en el buffer
            agent.memory.push(state, action, reward, next_state, done)
            
            # Actualizamos el agente con muestreo diversificado solo cada UPDATE_FREQUENCY pasos
            if len(agent.memory) > BATCH_SIZE and step % UPDATE_FREQUENCY == 0:
                # Reemplazamos el muestreo estándar con nuestro muestreo diversificado
                batch = diversified_sample_from_buffer(agent.memory.buffer, BATCH_SIZE, agent.device)
                
                if batch is not None:
                    state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch
                    
                    # Lógica de actualización similar a la del agente
                    with torch.no_grad():
                        next_action = agent.actor_target(next_state_batch)
                        target_q = agent.critic_target(next_state_batch, next_action)
                        target_value = reward_batch + (1 - done_batch) * agent.gamma * target_q
                    
                    # Valor Q actual
                    current_q = agent.critic(state_batch, action_batch)
                    
                    # Pérdida del crítico (MSE)
                    critic_loss = F.mse_loss(current_q, target_value)
                    
                    # Optimizamos el crítico
                    agent.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1.0)
                    agent.critic_optimizer.step()
                    
                    # Actualizamos el actor (menos frecuentemente que el crítico para mayor estabilidad)
                    if step % (UPDATE_FREQUENCY * 2) == 0:
                        actor_action = agent.actor(state_batch)
                        actor_loss = -agent.critic(state_batch, actor_action).mean()
                        
                        # Optimizamos el actor
                        agent.actor_optimizer.zero_grad()
                        actor_loss.backward()
                        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 1.0)
                        agent.actor_optimizer.step()
                        
                        # Actualizamos suavemente las redes objetivo
                        for target_param, param in zip(agent.actor_target.parameters(), agent.actor.parameters()):
                            target_param.data.copy_(agent.tau * param.data + (1 - agent.tau) * target_param.data)
                        
                        for target_param, param in zip(agent.critic_target.parameters(), agent.critic.parameters()):
                            target_param.data.copy_(agent.tau * param.data + (1 - agent.tau) * target_param.data)
                        
                        # Almacenar valores de pérdida del actor
                        actor_loss_value = actor_loss.item()
                        actor_losses.append(actor_loss_value)
                        
                        # Liberar tensor del actor
                        del actor_loss
                    
                    # Almacenar valores de pérdida del crítico
                    critic_loss_value = critic_loss.item()
                    critic_losses.append(critic_loss_value)
                    
                    # Liberar tensores explícitamente
                    del critic_loss
                    del state_batch, action_batch, reward_batch, next_state_batch, done_batch
                    
                    # Forzar limpieza de memoria cada 100 pasos
                    if step % 100 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # Actualizamos el estado y la recompensa
            state = next_state
            episode_reward += reward
            step += 1
            
            progress_bar.update(1)
        
        progress_bar.close()
        
        # Calcular velocidad de entrenamiento
        episode_duration = time.time() - episode_start_time
        iterations_per_second = step / episode_duration if episode_duration > 0 else 0
        training_speeds.append(iterations_per_second)
        
        print(f"Velocidad de entrenamiento: {iterations_per_second:.2f} it/s")
        
        # Ajustar frecuencia de actualización basada en la velocidad de entrenamiento
        if episode > 10 and len(training_speeds) > 5:
            avg_initial_speed = sum(training_speeds[:5]) / 5
            current_speed_ratio = iterations_per_second / avg_initial_speed
            
            if current_speed_ratio < 0.6:
                print("\n¡Alerta! Desaceleración detectada. Realizando limpieza agresiva...")
                
                # Reconstruir completamente el buffer (mantener solo 30% de las experiencias más recientes)
                if isinstance(agent.memory, MemoryEfficientReplayBuffer):
                    print(f"Reconstruyendo buffer de experiencias...")
                    old_buffer = agent.memory.buffer
                    keep_count = int(len(old_buffer) * 0.3)
                    new_buffer = old_buffer[-keep_count:]
                    
                    # Crear un nuevo buffer desde cero
                    agent.memory = MemoryEfficientReplayBuffer(BUFFER_CAPACITY, agent.device)
                    agent.memory.buffer = new_buffer
                    agent.memory.position = len(new_buffer) % agent.memory.capacity
                    
                    # Eliminar old_buffer explícitamente
                    del old_buffer
                
                # Aumentar frecuencia de actualización para reducir computación
                UPDATE_FREQUENCY = min(5, UPDATE_FREQUENCY + 1)
                print(f"Ajustando frecuencia de actualización a 1 cada {UPDATE_FREQUENCY} pasos")
                
                # Forzar sincronización CUDA y limpieza completa
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Forzar limpieza completa de memoria
                clear_memory()
        
        # Reducir el ruido para exploración
        agent.noise.decay_sigma()
        
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
        
        # Comprobar memoria cada pocos episodios
        if episode % MEMORY_CHECK_INTERVAL == 0:
            ram, gpu = monitor_memory()
            print(f"\nMemoria actual - RAM: {ram:.2f} MB, GPU: {gpu}")
            
            # Si el uso de memoria es alto, realizar limpieza
            if ram > MEMORY_CLEANUP_THRESHOLD:
                print("Detectado uso elevado de memoria, realizando limpieza...")
                clear_memory()
        
        # Limpieza periódica del buffer de experiencias (cada 30 episodios en lugar de 50)
        if episode % 30 == 0 and episode > 0:
            print("\nRealizando limpieza del buffer de experiencias...")
            if isinstance(agent.memory, MemoryEfficientReplayBuffer):
                if len(agent.memory) > BATCH_SIZE * 10:  # Mantener un mínimo de experiencias
                    print(f"Tamaño del buffer antes de limpieza: {len(agent.memory)}")
                    # Reducir la fracción a mantener para una limpieza más agresiva
                    agent.memory.clear_old_experiences(keep_fraction=0.4)  # Mantener el 40% más reciente
                    print(f"Tamaño del buffer después de limpieza: {len(agent.memory)}")
            # Forzar sincronización de GPU
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            clear_memory()
        
        # Evaluamos el agente periódicamente
        if episode % EVAL_EVERY == 0 or episode == NUM_EPISODES - 1:
            # Reinicio periódico de capas de red para escapar de mínimos locales
            if episode % 100 == 0 and episode > 0:
                print("\nReiniciando parcialmente la red del actor para escapar del mínimo local...")
                # Reiniciar solo las últimas capas del actor
                nn.init.xavier_uniform_(agent.actor.fc3.weight)
                nn.init.zeros_(agent.actor.fc3.bias)
                
                # Opcional: También podemos reiniciar parcialmente el crítico
                nn.init.xavier_uniform_(agent.critic.fc3.weight)
                nn.init.zeros_(agent.critic.fc3.bias)
                
                # Y ajustar la tasa de aprendizaje
                for param_group in agent.actor_optimizer.param_groups:
                    original_lr = ACTOR_LR  # Usar la constante original
                    current_lr = param_group['lr']
                    # Reiniciar al 80% de la tasa original si ha decaído demasiado
                    if current_lr < original_lr * 0.2:
                        param_group['lr'] = original_lr * 0.8
                        print(f"Restaurando tasa de aprendizaje del actor a {param_group['lr']:.6f}")
                
                # Limpiar memoria intensivamente antes de continuar
                clear_memory()
            
            print(f"\nEvaluando agente en episodio {episode+1}...")
            
            # Liberar memoria antes de la evaluación
            clear_memory()
            
            # Usar torch.no_grad() para la evaluación
            with torch.no_grad():
                eval_results = evaluate_agent(agent, env, EVAL_EPISODES)
            
            # Liberar caché de cuda de forma agresiva
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Guardar el ratio de Sharpe
            eval_sharpe_ratios.append(eval_results['sharpe_ratio'])
            
            print(f"Episodio {episode+1}/{NUM_EPISODES} | "
                  f"Tiempo: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d} | "
                  f"Recompensa: {episode_reward:.4f} | "
                  f"Balance: ${env.balance:.2f} | "
                  f"Crítico Loss: {np.mean(critic_losses) if critic_losses else 'N/A':.6f} | "
                  f"Actor Loss: {np.mean(actor_losses) if actor_losses else 'N/A':.6f} | "
                  f"Eval Balance: ${eval_results['avg_balance']:.2f} | "
                  f"Sharpe: {eval_results['sharpe_ratio']:.4f}")
            
            # Guardamos el mejor modelo según el ratio de Sharpe
            if eval_results['sharpe_ratio'] > best_eval_sharpe:
                best_eval_sharpe = eval_results['sharpe_ratio']
                best_eval_balance = eval_results['avg_balance']
                agent.save(os.path.join(model_dir, 'best_model.pth'))
                print(f"Nuevo mejor modelo guardado con Sharpe: {best_eval_sharpe:.4f}, Balance: ${best_eval_balance:.2f}")
                
                # Forzar limpieza de memoria después de guardar
                clear_memory()
            
            # Comprobar early stopping
            if early_stopping(eval_results['sharpe_ratio'], eval_results['avg_balance']):
                print(f"\nEarly stopping activado después de {episode+1} episodios sin mejora")
                break
        else:
            print(f"Episodio {episode+1}/{NUM_EPISODES} | "
                  f"Tiempo: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d} | "
                  f"Recompensa: {episode_reward:.4f} | "
                  f"Balance: ${env.balance:.2f} | "
                  f"Crítico Loss: {np.mean(critic_losses) if critic_losses else 'N/A':.6f} | "
                  f"Actor Loss: {np.mean(actor_losses) if actor_losses else 'N/A':.6f}")
        
        # Guardamos el modelo periódicamente
        if (episode + 1) % SAVE_MODEL_EVERY == 0:
            # Limpiar memoria antes de guardar
            clear_memory()
            agent.save(os.path.join(model_dir, f'model_ep{episode+1}.pth'))
            print(f"Modelo del episodio {episode+1} guardado")
    
    # Guardamos el modelo final
    clear_memory()  # Limpiar memoria antes de guardar
    agent.save(os.path.join(model_dir, 'final_model.pth'))
    
    # Guardamos las métricas de entrenamiento
    training_metrics = {
        'rewards': episode_rewards,
        'balances': episode_balances,
        'critic_losses': episode_critic_losses,
        'actor_losses': episode_actor_losses,
        'sharpe_ratios': eval_sharpe_ratios,
        'best_balance': best_eval_balance,
        'best_sharpe': best_eval_sharpe,
        'training_duration': time.time() - start_time,
        'training_speeds': training_speeds  # Guardamos también los datos de velocidad
    }
    
    save_training_metrics(training_metrics, model_dir)
    
    # Graficamos los resultados
    plot_training_results(
        episode_rewards, 
        episode_balances, 
        episode_critic_losses, 
        episode_actor_losses,
        eval_sharpe_ratios,
        model_dir
    )
    
    # Evaluación final más detallada
    print("\n" + "="*50)
    print("Evaluación final del modelo")
    print("="*50 + "\n")
    
    # Limpiar memoria antes de la evaluación final
    clear_memory()
    
    # Usar torch.no_grad para evaluación final
    with torch.no_grad():
        final_eval = evaluate_agent(agent, env, EVAL_EPISODES * 2, render=True)
    
    print("\nResultados de la evaluación final:")
    print(f"Recompensa promedio: {final_eval['avg_reward']:.4f}")
    print(f"Balance promedio: ${final_eval['avg_balance']:.2f}")
    print(f"Mejor balance: ${max(final_eval['final_balances']):.2f}")
    print(f"Peor balance: ${min(final_eval['final_balances']):.2f}")
    print(f"Ratio Sharpe: {final_eval['sharpe_ratio']:.4f}")
    
    # Guardamos las métricas de evaluación
    evaluation_metrics = {
        'avg_reward': final_eval['avg_reward'],
        'avg_balance': final_eval['avg_balance'],
        'final_balances': final_eval['final_balances'],
        'best_balance': max(final_eval['final_balances']),
        'worst_balance': min(final_eval['final_balances']),
        'sharpe_ratio': final_eval['sharpe_ratio']
    }
    
    with open(os.path.join(model_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(evaluation_metrics, f, indent=4)
    
    print(f"\nEvaluación completada. Resultados guardados en {model_dir}")
    print(f"Mejor modelo guardado en: {os.path.join(model_dir, 'best_model.pth')}")
    
    # Limpieza final de memoria
    print("\nRealizando limpieza final de memoria...")
    
    # Liberar referencias al buffer de memoria
    agent.memory.buffer.clear()
    del agent.memory.buffer
    
    # Liberar dispositivo CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Forzar recolección de basura
    gc.collect()
    
    # Comprobación final de memoria
    ram, gpu = monitor_memory()
    print(f"Memoria final - RAM: {ram:.2f} MB, GPU: {gpu}")
    print("Entrenamiento completado y memoria liberada")
    
    return os.path.join(model_dir, 'best_model.pth')

if __name__ == "__main__":
    main()