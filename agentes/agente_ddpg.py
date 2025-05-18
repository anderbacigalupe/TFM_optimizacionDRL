import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import copy

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, min_weight=0.05):  # Aumentado a 128
        super(Actor, self).__init__()
        self.min_weight = min_weight
        self.action_dim = action_dim
        
        # Añadimos normalización de capas
        self.layer_norm_input = nn.LayerNorm(state_dim)
        
        # Red más profunda con capas más amplias
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # Inicialización de pesos para mejorar convergencia
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)
    
    def forward(self, state):
        x = self.layer_norm_input(state)
        x = F.leaky_relu(self.fc1(x))
        x = self.layer_norm1(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.layer_norm2(x)
        
        # Aplicamos softmax para asegurar que los pesos sumen 1
        weights = F.softmax(self.fc3(x), dim=-1)
        
        # Aplicamos el peso mínimo por activo
        if self.min_weight > 0:
            weights = self._apply_min_weight_constraint(weights)
        
        return weights
    
    def _apply_min_weight_constraint(self, weights, max_iterations=10):
        """
        Aplica restricción de peso mínimo mediante proyección iterativa.
        Este método es más preciso que el simple clamp.
        """
        batch_size = weights.shape[0]
        for _ in range(max_iterations):
            # Identificar activos con peso insuficiente
            below_min = weights < self.min_weight
            if not torch.any(below_min):
                break
                
            # Calcular déficit total
            deficit = torch.sum((self.min_weight - weights) * below_min, dim=1, keepdim=True)
            
            # Identificar activos que pueden reducir su peso
            above_min = weights > self.min_weight
            excess_weight = torch.sum((weights - self.min_weight) * above_min, dim=1, keepdim=True)
            
            # Redistribuir proporcionalmente
            if torch.all(excess_weight > 0):
                reduction_factor = deficit / excess_weight
                reduction = (weights - self.min_weight) * above_min * reduction_factor
                weights = weights - reduction
                weights = torch.where(below_min, torch.ones_like(weights) * self.min_weight, weights)
        
        # Normalizar para asegurar que sumen 1
        weights = weights / torch.sum(weights, dim=1, keepdim=True)
        return weights


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):  # Aumentado a 128
        super(Critic, self).__init__()
        # Normalización para entradas
        self.layer_norm_state = nn.LayerNorm(state_dim)
        
        # Primera rama procesa solo el estado
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        
        # Segunda capa combina estado procesado con acción
        self.fc2 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # Capa adicional para mayor capacidad
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.layer_norm3 = nn.LayerNorm(hidden_dim // 2)
        
        # Capa de salida: valor Q
        self.fc4 = nn.Linear(hidden_dim // 2, 1)
        
        # Inicialización de pesos
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
        nn.init.zeros_(self.fc4.bias)
    
    def forward(self, state, action):
        # Normalización de entradas
        state = self.layer_norm_state(state)
        
        # Procesamos el estado
        state_value = F.leaky_relu(self.fc1(state))
        state_value = self.layer_norm1(state_value)
        
        # Concatenamos el estado procesado con la acción
        x = torch.cat([state_value, action], dim=1)
        x = F.leaky_relu(self.fc2(x))
        x = self.layer_norm2(x)
        
        # Capa adicional
        x = F.leaky_relu(self.fc3(x))
        x = self.layer_norm3(x)
        
        # Valor Q
        q_value = self.fc4(x)
        return q_value


class PrioritizedReplayBuffer:
    def __init__(self, capacity, device, alpha=0.6, beta_start=0.4, beta_end=1.0, beta_annealing_steps=1000):
        self.buffer = []
        self.capacity = capacity
        self.device = device
        self.position = 0
        self.alpha = alpha  # Determina la importancia del TD-error
        self.beta = beta_start  # Para corrección de importance-sampling
        self.beta_increment = (beta_end - beta_start) / beta_annealing_steps
        
        # Inicializar prioridades
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0  # Prioridad máxima inicial
    
    def push(self, state, action, reward, next_state, done):
        """Añade una nueva experiencia al buffer con máxima prioridad"""
        experience = (state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        # Asignar máxima prioridad a nueva experiencia
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """Muestrea experiencias basado en sus prioridades"""
        buffer_len = len(self.buffer)
        if buffer_len < batch_size:
            indices = np.random.choice(buffer_len, batch_size, replace=True)
        else:
            # Calcular probabilidades basadas en prioridades
            probs = self.priorities[:buffer_len] ** self.alpha
            probs /= probs.sum()
            
            # Muestrear según probabilidades
            indices = np.random.choice(buffer_len, batch_size, replace=False, p=probs)
        
        # Obtener experiencias y calcular weights para importance sampling
        samples = [self.buffer[idx] for idx in indices]
        weights = (buffer_len * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalizar a 1
        
        # Incrementar beta para convergencia
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Convertir a arrays
        states, actions, rewards, next_states, dones = zip(*samples)
        
        # Convertir a tensores y mover a GPU
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)
        
        return indices, states, actions, rewards, next_states, dones, weights
    
    def update_priorities(self, indices, td_errors):
        """Actualiza las prioridades basadas en el TD-error"""
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-5  # Evita prioridad 0
            self.max_priority = max(self.max_priority, self.priorities[idx])
    
    def __len__(self):
        return len(self.buffer)


class OUNoise:
    """
    Ornstein-Uhlenbeck process para exploración con ruido correlacionado,
    con decaimiento gradual del ruido.
    """
    def __init__(self, action_dim, mu=0, theta=0.15, sigma_start=0.2, sigma_end=0.05, decay_rate=0.995):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma_start
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.decay_rate = decay_rate
        self.reset()
    
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
    
    def decay_sigma(self):
        """Reduce gradualmente el valor de sigma."""
        self.sigma = max(self.sigma_end, self.sigma * self.decay_rate)
    
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state


class EarlyStopping:
    """Clase para implementar early stopping durante el entrenamiento"""
    def __init__(self, patience=20, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_sharpe = -float('inf')
        self.best_return = -float('inf')
        self.early_stop = False
    
    def __call__(self, sharpe_ratio, portfolio_return):
        # Mejora si alguna de las métricas principales mejora significativamente
        improvement = False
        
        if sharpe_ratio > self.best_sharpe + self.min_delta:
            self.best_sharpe = sharpe_ratio
            improvement = True
        
        if portfolio_return > self.best_return + self.min_delta and sharpe_ratio >= self.best_sharpe * 0.95:
            self.best_return = portfolio_return
            improvement = True
        
        if improvement:
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class DDPGAgent:
    def __init__(
        self, 
        state_dim, 
        action_dim, 
        hidden_dim=128,  # Aumentado a 128
        actor_lr=0.0005,  # Reducido para mayor estabilidad
        critic_lr=0.001, 
        gamma=0.99, 
        tau=0.005, 
        buffer_capacity=500000,
        batch_size=128,
        min_weight=0.05,
        noise_sigma_start=0.2,
        noise_sigma_end=0.05,
        noise_decay=0.995,
        prioritized_replay=True,
        n_step_returns=3,  # Para multi-step returns
        l2_reg=1e-4  # Regularización L2
    ):
        # Configuración del dispositivo (GPU o CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando dispositivo: {self.device}")
        
        # Parámetros
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.n_step_returns = n_step_returns
        self.min_weight = min_weight
        
        # Redes: actor y crítico
        self.actor = Actor(state_dim, action_dim, hidden_dim, min_weight).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        
        # Optimizador con regularización L2
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), 
            lr=actor_lr, 
            weight_decay=l2_reg
        )
        
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), 
            lr=critic_lr, 
            weight_decay=l2_reg
        )
        
        # Buffer de experiencia
        if prioritized_replay:
            self.memory = PrioritizedReplayBuffer(buffer_capacity, self.device)
            self.prioritized_replay = True
        else:
            self.memory = deque(maxlen=buffer_capacity)
            self.prioritized_replay = False
        
        # Ruido para exploración
        self.noise = OUNoise(
            action_dim, 
            sigma_start=noise_sigma_start,
            sigma_end=noise_sigma_end,
            decay_rate=noise_decay
        )
        
        # Cola para n-step returns
        self.n_step_buffer = deque(maxlen=n_step_returns)
        
        # Tracking de rendimiento para early stopping
        self.early_stopping = EarlyStopping(patience=50)
        
        # Modo de entrenamiento/evaluación
        self.training_mode = True
        
        # Historial de métricas
        self.metrics_history = {
            'critic_loss': [],
            'actor_loss': [],
            'avg_reward': [],
            'sharpe_ratio': [],
            'portfolio_value': []
        }
    
    def select_action(self, state, add_noise=True):
        """Selecciona una acción basada en el estado actual."""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state).squeeze(0).cpu().numpy()
            
        if add_noise and self.training_mode:
            noise = self.noise.sample()
            action = np.clip(action + noise, 0, 1)
            
            # Normalizar para que los pesos sumen 1
            if np.sum(action) > 0:
                action = action / np.sum(action)
        
        return action
    
    def _compute_n_step_returns(self, n_step_buffer, gamma):
        """Calcula los retornos de n pasos para una mejor estimación del valor."""
        reward = 0
        for i, (_, _, r, _, _) in enumerate(n_step_buffer):
            reward += (gamma ** i) * r
        
        state, action, _, next_state, done = n_step_buffer[-1]
        return state, action, reward, next_state, done
    
    def push_experience(self, state, action, reward, next_state, done):
        """Añade experiencia al buffer, con soporte para n-step returns."""
        # Guardar la experiencia en el buffer n-step
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        # Si el buffer n-step está lleno, calcular retorno n-step y guardar
        if len(self.n_step_buffer) == self.n_step_returns:
            n_state, n_action, n_reward, n_next_state, n_done = self._compute_n_step_returns(
                self.n_step_buffer, self.gamma
            )
            
            # Guardar en el buffer principal
            if self.prioritized_replay:
                self.memory.push(n_state, n_action, n_reward, n_next_state, n_done)
            else:
                self.memory.append((n_state, n_action, n_reward, n_next_state, n_done))
    
    def update(self):
        """Actualiza las redes de actor y crítico usando un batch de experiencias."""
        if self.prioritized_replay:
            if len(self.memory) < self.batch_size:
                return None, None
                
            # Muestrear con prioridad
            indices, state, action, reward, next_state, done, weights = self.memory.sample(self.batch_size)
        else:
            if len(self.memory) < self.batch_size:
                return None, None
                
            # Muestreo aleatorio
            batch = random.sample(self.memory, self.batch_size)
            state, action, reward, next_state, done = zip(*batch)
            
            # Convertir a tensores
            state = torch.FloatTensor(np.array(state)).to(self.device)
            action = torch.FloatTensor(np.array(action)).to(self.device)
            reward = torch.FloatTensor(np.array(reward)).unsqueeze(1).to(self.device)
            next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
            done = torch.FloatTensor(np.array(done)).unsqueeze(1).to(self.device)
            weights = torch.ones_like(reward).to(self.device)  # Sin importance sampling
        
        # Actualizar el crítico
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_q = self.critic_target(next_state, next_action)
            target_value = reward + (1 - done) * (self.gamma ** self.n_step_returns) * target_q
        
        # Valor Q actual
        current_q = self.critic(state, action)
        
        # TD-error para actualizar prioridades
        td_errors = torch.abs(target_value - current_q).detach().cpu().numpy()
        
        # Pérdida del crítico con importance sampling
        critic_loss = (weights * F.mse_loss(current_q, target_value, reduction='none')).mean()
        
        # Optimizamos el crítico
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Actualizamos el actor
        actor_action = self.actor(state)
        actor_loss = -(weights * self.critic(state, actor_action)).mean()
        
        # Optimizamos el actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Actualizamos prioridades en el buffer
        if self.prioritized_replay:
            self.memory.update_priorities(indices, td_errors.flatten())
        
        # Actualizamos las redes objetivo (soft update)
        self._update_target_networks()
        
        # Guardamos métricas
        self.metrics_history['critic_loss'].append(critic_loss.item())
        self.metrics_history['actor_loss'].append(actor_loss.item())
        
        return critic_loss.item(), actor_loss.item()
    
    def _update_target_networks(self):
        """Actualiza suavemente las redes objetivo con los pesos de las redes principales."""
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def decay_exploration(self):
        """Reduce gradualmente el ruido para exploración."""
        self.noise.decay_sigma()
    
    def check_early_stopping(self, sharpe_ratio, portfolio_return):
        """Comprueba si debe detenerse el entrenamiento basado en métricas de rendimiento."""
        return self.early_stopping(sharpe_ratio, portfolio_return)
    
    def save(self, path):
        """Guarda los pesos de las redes del agente y otros parámetros importantes."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'metrics_history': self.metrics_history,
            'hidden_dim': self.hidden_dim,
            'prioritized_replay': self.prioritized_replay,
            'n_step_returns': self.n_step_returns,
            'min_weight': self.min_weight,
        }, path)
        print(f"Modelo guardado en {path}")
    
    def load(self, path):
        """Carga los pesos de las redes del agente y restaura otros parámetros."""
        map_location = self.device
        checkpoint = torch.load(path, map_location=map_location)
        
        # Cargar configuración si está disponible
        if 'hidden_dim' in checkpoint:
            if checkpoint['hidden_dim'] != self.hidden_dim:
                print(f"Advertencia: El modelo guardado tiene hidden_dim={checkpoint['hidden_dim']}, pero el actual es {self.hidden_dim}")
                self.hidden_dim = checkpoint['hidden_dim']
                # Recrear redes con la dimensión correcta
                self.actor = Actor(self.state_dim, self.action_dim, self.hidden_dim, self.min_weight).to(self.device)
                self.actor_target = copy.deepcopy(self.actor).to(self.device)
                self.critic = Critic(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
                self.critic_target = copy.deepcopy(self.critic).to(self.device)
        
        # Cargar estados de los modelos
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        
        # Cargar estados de los optimizadores
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        # Cargar historial de métricas si existe
        if 'metrics_history' in checkpoint:
            self.metrics_history = checkpoint['metrics_history']
        
        # Restaurar otros parámetros
        if 'n_step_returns' in checkpoint:
            self.n_step_returns = checkpoint['n_step_returns']
        
        if 'min_weight' in checkpoint:
            self.min_weight = checkpoint['min_weight']
            self.actor.min_weight = self.min_weight
        
        print(f"Modelo cargado desde {path}")
    
    def train(self):
        """Establece el agente en modo entrenamiento."""
        self.training_mode = True
        self.actor.train()
        self.critic.train()
        self.actor_target.train()
        self.critic_target.train()
    
    def eval(self):
        """Establece el agente en modo evaluación."""
        self.training_mode = False
        self.actor.eval()
        self.critic.eval()
        self.actor_target.eval()
        self.critic_target.eval()
    
    def get_metrics_history(self):
        """Devuelve el historial de métricas para análisis y visualización."""
        return self.metrics_history