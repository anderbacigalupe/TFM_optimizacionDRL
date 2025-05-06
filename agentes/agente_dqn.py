import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import gymnasium as gym

class DQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, action_dim, n_discrete_bins=10, learning_rate=1e-3, 
                 gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 buffer_capacity=10000, batch_size=64):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_discrete_bins = n_discrete_bins  # Número de bins para discretizar cada dimensión
        
        # Calculamos el número total de acciones discretas
        # Para un portafolio con n activos, cada uno con m valores posibles
        self.total_discrete_actions = n_discrete_bins ** action_dim
        
        # Generamos todas las combinaciones posibles de acciones discretas
        self.discrete_actions = self._generate_discrete_actions()
        
        # Parámetros de DQN
        self.gamma = gamma  # Factor de descuento
        self.epsilon = epsilon_start  # Exploración inicial
        self.epsilon_end = epsilon_end  # Exploración mínima
        self.epsilon_decay = epsilon_decay  # Tasa de decaimiento de exploración
        self.batch_size = batch_size
        
        # Redes neuronales: principal y objetivo
        self.policy_net = DQNNetwork(state_dim, self.total_discrete_actions)
        self.target_net = DQNNetwork(state_dim, self.total_discrete_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Modo evaluación (no entrena)
        
        # Optimizador
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Buffer de experiencia para replay
        self.memory = ReplayBuffer(buffer_capacity)
    
    def _generate_discrete_actions(self):
        """
        Genera todas las combinaciones posibles de acciones discretas.
        Cada acción debe sumar 1 (distribución de pesos).
        """
        # Limitamos el número de combinaciones para evitar explosión combinatoria
        if self.action_dim > 5 and self.n_discrete_bins > 5:
            print("Advertencia: Demasiadas combinaciones posibles. Reduciendo bins a 5.")
            self.n_discrete_bins = 5
        
        # Generamos las combinaciones de manera más eficiente para portafolios
        discrete_values = np.linspace(0, 1, self.n_discrete_bins)
        
        # Para asegurar que cada combinación sume 1, generamos todas las particiones posibles
        actions = []
        
        # Usamos un enfoque recursivo para generar combinaciones que sumen 1
        def generate_combinations(remaining_assets, remaining_value=1.0, current_combo=[]):
            if remaining_assets == 1:
                # Último activo recibe el valor restante
                actions.append(current_combo + [remaining_value])
                return
                
            # Para cada activo excepto el último, probamos diferentes valores
            for value in discrete_values:
                if value <= remaining_value:
                    generate_combinations(remaining_assets-1, remaining_value-value, current_combo + [value])
        
        # Limitamos el número de activos para la recursión
        max_assets_for_recursion = 4
        
        if self.action_dim <= max_assets_for_recursion:
            # Usar método recursivo para pocos activos
            generate_combinations(self.action_dim)
        else:
            # Para muchos activos, usamos un enfoque más simple pero menos preciso
            # Generamos pesos aleatorios y los normalizamos
            print(f"Generando {min(1000, self.total_discrete_actions)} acciones aleatorias para {self.action_dim} activos")
            num_samples = min(1000, self.total_discrete_actions)  # Limitamos a 1000 combinaciones
            
            for _ in range(num_samples):
                weights = np.random.rand(self.action_dim)
                weights = weights / np.sum(weights)  # Normalizar para que sumen 1
                actions.append(weights.tolist())
        
        return np.array(actions)
    
    def select_action(self, state, training=True):
        """
        Selecciona una acción siguiendo una política epsilon-greedy.
        """
        if training and random.random() < self.epsilon:
            # Exploración: acción aleatoria
            action_idx = random.randrange(len(self.discrete_actions))
        else:
            # Explotación: mejor acción según la red
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                action_idx = q_values.max(1)[1].item()
        
        # Devolvemos la acción continua correspondiente
        return self.discrete_actions[action_idx]
    
    def update_epsilon(self):
        """
        Actualiza el valor de epsilon para la política epsilon-greedy.
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def update(self):
        """
        Actualiza la red de política utilizando experiencias almacenadas en el buffer.
        """
        if len(self.memory) < self.batch_size:
            return
        
        # Muestreamos un batch de experiencias
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convertimos a tensores
        state_batch = torch.FloatTensor(np.array(states))
        action_indices = []
        
        # Encontramos los índices de las acciones en el array de acciones discretas
        for action in actions:
            # Buscamos la acción más cercana en nuestro conjunto discreto
            distances = np.sum((self.discrete_actions - action) ** 2, axis=1)
            closest_idx = np.argmin(distances)
            action_indices.append(closest_idx)
        
        action_batch = torch.LongTensor(action_indices)
        reward_batch = torch.FloatTensor(rewards)
        next_state_batch = torch.FloatTensor(np.array(next_states))
        done_batch = torch.FloatTensor(dones)
        
        # Calculamos los Q-values actuales
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Calculamos los Q-values objetivo
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            expected_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        
        # Calculamos la pérdida
        loss = nn.MSELoss()(current_q_values.squeeze(), expected_q_values)
        
        # Optimizamos
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping para estabilidad
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """
        Actualiza la red objetivo con los pesos de la red de política.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, path):
        """
        Guarda el modelo en disco.
        """
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'discrete_actions': self.discrete_actions,
        }, path)
    
    def load(self, path):
        """
        Carga el modelo desde disco.
        """
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.discrete_actions = checkpoint['discrete_actions']