import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PortfolioEnv(gym.Env):
    """Entorno personalizado para optimización de carteras compatible con Gymnasium."""

    def __init__(self, data, initial_balance=1_000_000):
        super().__init__()
        self.data = data
        self.initial_balance = initial_balance
        self.n_assets = data.shape[1]

        # Espacio de observación: precios actuales de los activos + cash disponible
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.n_assets + 1,), dtype=np.float32)

        # Espacio de acción: proporción de la cartera para cada activo (deben sumar 1)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        # Inicialmente todo el balance está en efectivo
        self.cash = self.initial_balance
        self.shares = np.zeros(self.n_assets)
        # Inicializamos los pesos del portafolio (teóricamente todos en efectivo)
        self.portfolio_weights = np.zeros(self.n_assets)
        self.done = False
        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        action = np.clip(action, 0, 1)  # Asegura que ninguna asignación sea negativa
        
        # Establece un peso mínimo para cada activo (por ejemplo, 0.05 o 5%)
        min_weight = 0.05  # Peso mínimo del 5%
        
        # Aplica el peso mínimo y normaliza
        if np.sum(action) > 0:  # Evita la división por cero
            # Primero, asegúrate de que cada activo tenga al menos el peso mínimo
            below_min = action < min_weight
            if np.any(below_min):
                # Asigna el peso mínimo a los activos por debajo del umbral
                action[below_min] = min_weight

        # Normaliza para que sumen 1
        total_weight = np.sum(action)
        
        if total_weight > 0:
            action = action / total_weight  # Normaliza solo si la suma no es cero
        
        # Verifica que los pesos sumen 1 y que cada activo tenga al menos el peso mínimo
        action = np.maximum(action, min_weight)  # Asegura que cada peso sea al menos el mínimo
        action = action / np.sum(action)  # Normaliza nuevamente después de aplicar el peso mínimo

        prev_prices = self.data[self.current_step]

        if self.current_step >= len(self.data):
            self.done = True
            return self._get_observation(), 0, self.done, False, {}
        
        self.current_step += 1
        new_prices = self.data[self.current_step]
        price_relatives = new_prices / prev_prices

        slippage = 0.001  # 0.1%

        # Valor total del portafolio (acciones + efectivo)
        total_portfolio_value = np.sum(self.shares * prev_prices) + self.cash
        
        # Valores de asignación deseados
        target_allocation_value = total_portfolio_value * action
        
        # Calculamos las acciones objetivo (redondeando a números enteros)
        target_shares = np.floor(target_allocation_value / prev_prices)
        target_shares = np.maximum(target_shares, 0)  # Evita valores negativos
        
        # Delta de acciones (compra/venta)
        delta_shares = target_shares - self.shares
        
        # Aplicamos precios efectivos con slippage
        effective_buy_prices = prev_prices * (1 + slippage)   # compras
        effective_sell_prices = prev_prices * (1 - slippage)  # ventas
        
        # Calculamos el valor efectivo de las transacciones
        buy_value = np.sum(np.where(delta_shares > 0, delta_shares * effective_buy_prices, 0))
        sell_value = np.sum(np.where(delta_shares < 0, -delta_shares * effective_sell_prices, 0))
        
        # Verificamos si hay suficiente efectivo para las compras
        if buy_value > self.cash + sell_value:
            # No hay suficiente efectivo, ajustamos las compras
            available_cash = self.cash + sell_value
            scale_factor = available_cash / buy_value if buy_value > 0 else 0
            
            # Ajustamos delta_shares solo para compras
            buy_delta_shares = np.where(delta_shares > 0, delta_shares, 0)
            adjusted_buy_delta_shares = np.floor(buy_delta_shares * scale_factor)
            
            # Recalculamos delta_shares
            delta_shares = np.where(delta_shares > 0, adjusted_buy_delta_shares, delta_shares)
            
            # Recalculamos buy_value
            buy_value = np.sum(np.where(delta_shares > 0, delta_shares * effective_buy_prices, 0))

        # Actualizamos el efectivo disponible
        self.cash = self.cash + sell_value - buy_value
        
        # Comisiones por operación (mínimo 0.35, máximo 1% del valor negociado)
        total_traded_value = buy_value + sell_value
        buy_sell_commissions = np.where(delta_shares != 0,
                                        np.maximum(0.35, np.minimum(0.0035 * np.abs(delta_shares), 
                                                                    0.01 * np.abs(delta_shares) * prev_prices)),
                                        0)
        total_commission = np.sum(buy_sell_commissions)
        self.cash -= total_commission  # Aplicamos las comisiones
        
        # Actualizamos las acciones después de las transacciones
        self.shares = np.maximum(self.shares + delta_shares, 0)  # Evita acciones negativas

        
        # Calculamos el nuevo valor de la cartera con los nuevos precios
        new_portfolio_value = np.sum(self.shares * new_prices) + self.cash
        
        # Calculamos el crecimiento
        portfolio_growth = new_portfolio_value / total_portfolio_value
        
        # Actualizamos el balance total
        self.balance = new_portfolio_value
        
        # Actualizamos los pesos del portafolio
        if self.balance > 0:
            self.portfolio_weights = (self.shares * new_prices) / self.balance
        else:
            self.portfolio_weights = np.zeros_like(self.portfolio_weights)
        
        reward = np.log(portfolio_growth)  # Recompensa basada en el log del crecimiento

        obs = self._get_observation()
        info = {
            "commission": total_commission, 
            "slippage": slippage,
            "cash": self.cash,
            "shares": self.shares,
            "portfolio_value": new_portfolio_value,
            "min_weight_applied": any(below_min) if np.sum(action) > 0 else False  # Información adicional
        }
        return obs, reward, self.done, False, info


    def _get_observation(self):
        # Usamos el valor explícito de cash
        return np.concatenate(([self.cash], self.data[self.current_step]))

    def render(self):
        asset_values = self.shares * self.data[self.current_step]
        total_asset_value = np.sum(asset_values)
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Cash: {self.cash:.2f}, Assets: {total_asset_value:.2f}")
        print(f"Shares: {self.shares}")
        print(f"Weights: {self.portfolio_weights}")
