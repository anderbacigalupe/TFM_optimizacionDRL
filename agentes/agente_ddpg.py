import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import copy

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64, min_weight=0.05):
        super(Actor, self).__init__()
        self.min_weight = min_weight
        self.action_dim = action_dim
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # Inicialización de pesos para mejorar convergencia
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # Aplicamos softmax para asegurar que los pesos sumen 1
        weights = F.softmax(self.fc3(x), dim=-1)
        
        # Aplicamos el peso mínimo por activo
        if self.min_weight > 0:
            # Método 1: Redistribución simple (puede no ser exactamente min_weight)
            weights = torch.clamp(weights, min=self.min_weight)
            weights = weights / torch.sum(weights, dim=-1, keepdim=True)
            
            # Método 2 (alternativo): Forzar min_weight exacto mediante proyección iterativa
            # Este método es más preciso pero computacionalmente más costoso
            # self._apply_min_weight_constraint(weights)
        
        return weights
    
    def _apply_min_weight_constraint(self, weights, max_iterations=10):
        """
        Aplica restricción de peso mínimo mediante proyección iterativa.
        Este método es una alternativa más precisa pero más costosa.
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
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Critic, self).__init__()
        # Primera capa procesa solo el estado
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        
        # Segunda capa combina estado procesado con acción
        self.fc2 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        
        # Capa de salida: valor Q
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # Inicialización de pesos
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)
    
    def forward(self, state, action):
        # Procesamos el estado
        state_value = F.relu(self.fc1(state))
        
        # Concatenamos el estado procesado con la acción
        x = torch.cat([state_value, action], dim=1)
        x = F.relu(self.fc2(x))
        
        # Valor Q
        q_value = self.fc3(x)
        return q_value


class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.FloatTensor(np.array(state)).to(self.device),
            torch.FloatTensor(np.array(action)).to(self.device),
            torch.FloatTensor(np.array(reward)).unsqueeze(1).to(self.device),
            torch.FloatTensor(np.array(next_state)).to(self.device),
            torch.FloatTensor(np.array(done)).unsqueeze(1).to(self.device)
        )
    
    def __len__(self):
        return len(self.buffer)


class OUNoise:
    """
    Ornstein-Uhlenbeck process para exploración con ruido correlacionado.
    """
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()
    
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
    
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state


class DDPGAgent:
    def __init__(
        self, 
        state_dim, 
        action_dim, 
        hidden_dim=64,
        actor_lr=0.0007, 
        critic_lr=0.001, 
        gamma=0.99, 
        tau=0.005, 
        buffer_capacity=1000000,
        batch_size=128,
        min_weight=0.05,
        noise_sigma=0.1
    ):
        # Configuración del dispositivo (GPU o CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando dispositivo: {self.device}")
        
        # Dimensiones
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Hiperparámetros
        self.gamma = gamma  # factor de descuento
        self.tau = tau      # factor para soft update
        self.batch_size = batch_size
        
        # Redes: actor y crítico (políticas y target) - movidas a GPU
        self.actor = Actor(state_dim, action_dim, hidden_dim, min_weight).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Buffer de experiencia - con soporte para GPU
        self.memory = ReplayBuffer(buffer_capacity, self.device)
        
        # Ruido para exploración
        self.noise = OUNoise(action_dim, sigma=noise_sigma)
        
        # Flag para modo de evaluación
        self.training_mode = True
    
    def select_action(self, state, add_noise=True):
        """
        Selecciona una acción basada en el estado actual.
        
        Args:
            state: Estado actual del entorno
            add_noise: Si es True, añade ruido para exploración
            
        Returns:
            action: Vector de pesos para la cartera
        """
        # Convertir state a tensor y moverlo a GPU
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Obtener acción del actor (en GPU)
            action = self.actor(state).squeeze(0)
            # Mover a CPU para procesamiento con numpy
            action = action.cpu().numpy()
            
        if add_noise and self.training_mode:
            noise = self.noise.sample()
            
            # Aplicamos el ruido y nos aseguramos de que los pesos sean válidos
            action = action + noise
            action = np.clip(action, 0, 1)  # Aseguramos que esté entre 0 y 1
            
            # Normalizamos para que sumen 1
            if np.sum(action) > 0:
                action = action / np.sum(action)
        
        return action
    
    def update(self):
        """
        Actualiza las redes de actor y crítico usando un batch de experiencias.
        
        Returns:
            critic_loss, actor_loss: Pérdidas del crítico y actor
        """
        if len(self.memory) < self.batch_size:
            return None, None
        
        # Muestreamos un batch del buffer de experiencia (ya en GPU)
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        
        # Actualizamos el crítico
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_q = self.critic_target(next_state, next_action)
            target_value = reward + (1 - done) * self.gamma * target_q
        
        # Valor Q actual
        current_q = self.critic(state, action)
        
        # Pérdida del crítico (MSE)
        critic_loss = F.mse_loss(current_q, target_value)
        
        # Optimizamos el crítico
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Añadimos gradient clipping para estabilidad
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Actualizamos el actor
        actor_action = self.actor(state)
        actor_loss = -self.critic(state, actor_action).mean()
        
        # Optimizamos el actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # Añadimos gradient clipping para estabilidad
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Actualizamos las redes objetivo (soft update)
        self._update_target_networks()
        
        return critic_loss.item(), actor_loss.item()
    
    def _update_target_networks(self):
        """
        Actualiza suavemente las redes objetivo con los pesos de las redes principales.
        """
        # Actualizar parámetros del actor objetivo
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # Actualizar parámetros del crítico objetivo
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, path):
        """
        Guarda los pesos de las redes del agente.
        """
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, path)
        print(f"Modelo guardado en {path}")
    
    def load(self, path):
        """
        Carga los pesos de las redes del agente.
        """
        # Cargar con map_location para manejar correctamente GPU/CPU
        map_location = self.device
        checkpoint = torch.load(path, map_location=map_location)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        print(f"Modelo cargado desde {path}")
    
    def train(self):
        """
        Establece el agente en modo entrenamiento.
        """
        self.training_mode = True
        self.actor.train()
        self.critic.train()
        self.actor_target.train()
        self.critic_target.train()
    
    def eval(self):
        """
        Establece el agente en modo evaluación.
        """
        self.training_mode = False
        self.actor.eval()
        self.critic.eval()
        self.actor_target.eval()
        self.critic_target.eval()