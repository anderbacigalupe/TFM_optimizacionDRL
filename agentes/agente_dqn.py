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
    
print("DEBUG: Loading AGENTE_DQN.PY - Version with min_weight and tau --")

class DQNAgent:
    def __init__(self, state_dim, action_dim, n_discrete_bins=10, learning_rate=1e-3, 
                 gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 buffer_capacity=10000, batch_size=64, min_weight=0.05, tau=0.005):
        
        # Configuración de dispositivo (GPU o CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando dispositivo: {self.device}")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_discrete_bins = n_discrete_bins  # Número de bins para discretizar cada dimensión
        self.min_weight = min_weight  # Peso mínimo para cada activo (5%)
        self.tau = tau  # Factor para soft update de la red objetivo
        
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
        
        # Redes neuronales: principal y objetivo (movidas a GPU)
        self.policy_net = DQNNetwork(state_dim, len(self.discrete_actions)).to(self.device)
        self.target_net = DQNNetwork(state_dim, len(self.discrete_actions)).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Modo evaluación (no entrena)
        
        # Optimizador
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Buffer de experiencia para replay
        self.memory = ReplayBuffer(buffer_capacity)
        
        # Inicializar diccionario para cachear los índices de acciones más cercanas
        self.action_cache = {}
        
        # Convertimos discrete_actions a tensor en GPU para cálculos más rápidos
        self.discrete_actions_tensor = torch.FloatTensor(self.discrete_actions).to(self.device)
    
    def _generate_discrete_actions(self):
        """
        Genera todas las combinaciones posibles de acciones discretas.
        Cada acción debe sumar 1 (distribución de pesos) y respetar el peso mínimo.
        """
        # Ajustamos los bins según el número de activos para evitar explosión combinatoria
        if self.action_dim > 5 and self.n_discrete_bins > 5:
            print(f"Advertencia: Demasiadas combinaciones posibles. Reduciendo bins a 5 para {self.action_dim} activos.")
            self.n_discrete_bins = 5
        elif self.action_dim > 4 and self.n_discrete_bins > 7:
            print(f"Advertencia: Reduciendo bins a 7 para {self.action_dim} activos.")
            self.n_discrete_bins = 7
        
        # Ajustamos los valores discretos, considerando el peso mínimo
        min_weight = self.min_weight
        
        # El máximo peso por activo está limitado por el hecho de que todos los demás 
        # activos deben tener al menos el peso mínimo
        max_weight = 1.0 - (self.action_dim - 1) * min_weight
        
        # Generamos valores discretos entre min_weight y max_weight
        discrete_values = np.linspace(min_weight, max_weight, self.n_discrete_bins)
        
        # Para asegurar que cada combinación sume 1, generamos todas las particiones posibles
        actions = []
        
        # Usamos un enfoque recursivo para generar combinaciones que sumen 1
        def generate_combinations(remaining_assets, remaining_value=1.0, current_combo=[]):
            if remaining_assets == 1:
                # Último activo recibe el valor restante
                # Verificamos que el último activo respete el peso mínimo
                if remaining_value >= min_weight:
                    actions.append(current_combo + [remaining_value])
                return
                
            # Para cada activo excepto el último, probamos diferentes valores
            for value in discrete_values:
                # Verificamos que respete el peso mínimo y que quede suficiente para los restantes
                if value <= remaining_value - (remaining_assets - 1) * min_weight:
                    generate_combinations(remaining_assets-1, remaining_value-value, current_combo + [value])
        
        # Limitamos el número de activos para la recursión
        max_assets_for_recursion = 4
        
        if self.action_dim <= max_assets_for_recursion:
            # Usar método recursivo para pocos activos
            print(f"Generando combinaciones de pesos para {self.action_dim} activos usando método recursivo...")
            generate_combinations(self.action_dim)
            print(f"Se generaron {len(actions)} combinaciones posibles.")
        else:
            # Para muchos activos, usamos un enfoque más simple
            # Generamos pesos aleatorios y los normalizamos, asegurando el peso mínimo
            print(f"Generando combinaciones de pesos para {self.action_dim} activos usando método aleatorio...")
            num_samples = min(2000, self.total_discrete_actions)  # Aumentamos a 2000 combinaciones
            
            for _ in range(num_samples):
                # Generamos pesos aleatorios para cada activo
                weights = np.random.uniform(min_weight, 1.0, self.action_dim)
                
                # Normalizamos para que sumen 1
                weights = weights / np.sum(weights)
                
                # Aseguramos que respeten el peso mínimo
                below_min = weights < min_weight
                if np.any(below_min):
                    # Calculamos cuánto necesitamos reasignar
                    deficit = min_weight * np.sum(below_min) - np.sum(weights[below_min])
                    
                    # Quitamos proporcionalmente de los activos por encima del mínimo
                    above_min = ~below_min
                    if np.any(above_min):
                        weights[above_min] -= deficit * weights[above_min] / np.sum(weights[above_min])
                        weights[below_min] = min_weight
                
                # Normalizamos nuevamente por seguridad
                weights = weights / np.sum(weights)
                
                actions.append(weights.tolist())
            
            print(f"Se generaron {len(actions)} combinaciones aleatorias.")
        
        # Eliminamos posibles duplicados
        actions_array = np.array(actions)
        unique_actions = []
        seen = set()
        
        for action in actions:
            # Convertimos a una representación de string para comparar
            action_key = tuple(np.round(action, 3))
            if action_key not in seen:
                seen.add(action_key)
                unique_actions.append(action)
        
        print(f"Después de eliminar duplicados: {len(unique_actions)} acciones únicas.")
        return np.array(unique_actions)
    
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
                # Pasamos el estado a GPU
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                action_idx = q_values.max(1)[1].item()
        
        # Devolvemos la acción continua correspondiente
        return self.discrete_actions[action_idx]
    
    def update_epsilon(self):
        """
        Actualiza el valor de epsilon para la política epsilon-greedy.
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def _find_closest_action_index(self, action):
        """
        Encuentra el índice de la acción discreta más cercana a la acción continua dada.
        Utiliza caché para mejorar el rendimiento.
        """
        # Convertimos la acción a una tupla para usarla como clave en el diccionario
        action_key = tuple(np.round(action, 4))
        
        # Si ya hemos calculado este índice antes, lo devolvemos directamente
        if action_key in self.action_cache:
            return self.action_cache[action_key]
        
        # Calculamos las distancias usando GPU para aceleración
        action_tensor = torch.FloatTensor(action).to(self.device)
        distances = torch.sum((self.discrete_actions_tensor - action_tensor) ** 2, dim=1)
        closest_idx = torch.argmin(distances).item()
        
        # Guardamos en la caché para futuros usos
        self.action_cache[action_key] = closest_idx
        
        return closest_idx
    
    def update(self):
        """
        Actualiza la red de política utilizando experiencias almacenadas en el buffer.
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Muestreamos un batch de experiencias
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convertimos a tensores y movemos a GPU
        state_batch = torch.FloatTensor(np.array(states)).to(self.device)
        action_indices = []
        
        # Encontramos los índices de las acciones en el array de acciones discretas
        for action in actions:
            # Usamos nuestra función optimizada para encontrar el índice
            closest_idx = self._find_closest_action_index(action)
            action_indices.append(closest_idx)
        
        action_batch = torch.LongTensor(action_indices).to(self.device)
        reward_batch = torch.FloatTensor(rewards).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(next_states)).to(self.device)
        done_batch = torch.FloatTensor(dones).to(self.device)
        
        # Calculamos los Q-values actuales
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Calculamos los Q-values objetivo usando Double DQN
        with torch.no_grad():
            # Seleccionamos la mejor acción según la red de política
            policy_argmax = self.policy_net(next_state_batch).max(1)[1].unsqueeze(1)
            
            # Evaluamos esa acción con la red objetivo
            next_q_values = self.target_net(next_state_batch).gather(1, policy_argmax).squeeze(1)
            
            expected_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        
        # Calculamos la pérdida de Huber (más robusta que MSE)
        loss = nn.SmoothL1Loss()(current_q_values.squeeze(), expected_q_values)
        
        # Optimizamos
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping para estabilidad
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self, use_soft_update=True):
        """
        Actualiza la red objetivo con los pesos de la red de política.
        
        Args:
            use_soft_update: Si es True, hace una actualización suave (soft update)
                            Si es False, hace una actualización completa (hard update)
        """
        if use_soft_update:
            # Soft update: τ*θ_policy + (1-τ)*θ_target
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(
                    self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
                )
        else:
            # Hard update: θ_target = θ_policy
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
            'epsilon': self.epsilon,
            'min_weight': self.min_weight
        }, path)
        print(f"Modelo guardado en {path}")
    
    def load(self, path):
        """
        Carga el modelo desde disco.
        """
        # Determinamos el dispositivo para cargar correctamente
        map_location = self.device
        
        checkpoint = torch.load(path, map_location=map_location)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.discrete_actions = checkpoint['discrete_actions']
        
        # También creamos el tensor de acciones para GPU
        self.discrete_actions_tensor = torch.FloatTensor(self.discrete_actions).to(self.device)
        
        # Cargar valores adicionales si están disponibles
        if 'epsilon' in checkpoint:
            self.epsilon = checkpoint['epsilon']
        if 'min_weight' in checkpoint:
            self.min_weight = checkpoint['min_weight']
            
        print(f"Modelo cargado desde {path}")
        print(f"Número de acciones discretas: {len(self.discrete_actions)}")
    
    def train(self):
        """Establece las redes en modo entrenamiento"""
        self.policy_net.train()
        self.target_net.train()
    
    def eval(self):
        """Establece las redes en modo evaluación"""
        self.policy_net.eval()
        self.target_net.eval()