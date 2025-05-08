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
        # Aseguramos que la acción sea un array de numpy para evitar problemas
        action = np.array(action, dtype=np.float32)
        
        # Prevenir NaN en la acción
        if np.any(np.isnan(action)):
            print("¡ADVERTENCIA! Acción con valores NaN recibida.")
            action = np.ones(self.n_assets) / self.n_assets  # Usar pesos iguales como fallback
        
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
        else:
            # Si todos los pesos son cero, usar pesos iguales
            action = np.ones(self.n_assets) / self.n_assets
        
        # Verifica que los pesos sumen 1 y que cada activo tenga al menos el peso mínimo
        action = np.maximum(action, min_weight)  # Asegura que cada peso sea al menos el mínimo
        action = action / np.sum(action)  # Normaliza nuevamente después de aplicar el peso mínimo

        # Obtener los precios actuales
        prev_prices = self.data[self.current_step]
        
        # Verifica si el siguiente paso excedería el límite de datos
        if self.current_step >= len(self.data) - 1:
            self.done = True
            # Usa los precios actuales para la observación final
            obs = self._get_observation()
            return obs, 0, self.done, False, {"portfolio_value": self.balance}
        
        # Incrementamos el paso
        self.current_step += 1
        
        # Obtener los nuevos precios
        new_prices = self.data[self.current_step]
        
        # Prevenir precios negativos o cero (aunque esto no debería ocurrir en datos reales)
        prev_prices = np.maximum(prev_prices, 1e-8)
        new_prices = np.maximum(new_prices, 1e-8)
        
        # Calcular los cambios relativos de precios
        price_relatives = new_prices / prev_prices

        # Parámetro de deslizamiento (slippage)
        slippage = 0.001  # 0.1%

        # Valor total del portafolio (acciones + efectivo)
        total_portfolio_value = np.sum(self.shares * prev_prices) + self.cash
        
        # Prevenir valores negativos o cero para el portafolio
        if total_portfolio_value <= 0:
            total_portfolio_value = 1.0  # Valor mínimo para evitar división por cero
            self.cash = 1.0
            self.shares = np.zeros(self.n_assets)
        
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
        buy_sell_commissions = np.zeros(self.n_assets)
        for i in range(self.n_assets):
            if delta_shares[i] != 0:
                commission = max(0.35, min(0.0035 * abs(delta_shares[i]), 0.01 * abs(delta_shares[i]) * prev_prices[i]))
                buy_sell_commissions[i] = commission
        
        total_commission = np.sum(buy_sell_commissions)
        self.cash -= total_commission  # Aplicamos las comisiones
        
        # Aseguramos que el efectivo no sea negativo
        if self.cash < 0:
            self.cash = 0
        
        # Actualizamos las acciones después de las transacciones
        self.shares = np.maximum(self.shares + delta_shares, 0)  # Evita acciones negativas
        
        # Calculamos el nuevo valor de la cartera con los nuevos precios
        new_portfolio_value = np.sum(self.shares * new_prices) + self.cash
        
        # Calculamos el crecimiento con protección contra división por cero
        if total_portfolio_value > 0:
            portfolio_growth = new_portfolio_value / total_portfolio_value
        else:
            portfolio_growth = 1.0  # Neutral growth if portfolio value was zero
        
        # Protección contra crecimiento negativo o cero
        if portfolio_growth <= 0:
            portfolio_growth = 1e-8  # Valor pequeño positivo
        
        # Actualizamos el balance total
        self.balance = new_portfolio_value
        
        # Actualizamos los pesos del portafolio con protección contra división por cero
        if self.balance > 0:
            self.portfolio_weights = (self.shares * new_prices) / self.balance
        else:
            # Si el balance es cero o negativo, usar pesos iguales
            self.portfolio_weights = np.ones(self.n_assets) / self.n_assets
        
        # Aseguramos que los pesos estén en el rango [0, 1]
        self.portfolio_weights = np.clip(self.portfolio_weights, 0, 1)
        
        # Normalizamos los pesos para que sumen 1
        sum_weights = np.sum(self.portfolio_weights)
        if sum_weights > 0:
            self.portfolio_weights = self.portfolio_weights / sum_weights
        
        # Recompensa basada en el log del crecimiento con protección contra NaN
        try:
            reward = np.log(portfolio_growth)
        except (ValueError, RuntimeWarning):
            # Si hay un problema con el logaritmo, usar un valor alternativo
            if portfolio_growth > 1:
                reward = 0.01  # Pequeña recompensa positiva
            else:
                reward = -0.01  # Pequeña penalización
        
        # Prevenir recompensas extremas
        reward = np.clip(reward, -1.0, 1.0)
        
        # Verificación final de NaN
        if (np.isnan(self.balance) or np.isnan(self.cash) or np.isnan(reward) or 
            np.any(np.isnan(self.shares)) or np.any(np.isnan(self.portfolio_weights))):
            print("¡ADVERTENCIA! Se detectaron valores NaN en step():")
            print(f"Current step: {self.current_step}")
            print(f"Balance: {self.balance}, Cash: {self.cash}, Reward: {reward}")
            print(f"Shares: {self.shares}")
            print(f"Portfolio weights: {self.portfolio_weights}")
            print(f"Prev prices: {prev_prices}")
            print(f"New prices: {new_prices}")
            print(f"Price relatives: {price_relatives}")
            
            # Reiniciar a valores seguros
            self.balance = self.initial_balance
            self.cash = self.initial_balance
            self.shares = np.zeros(self.n_assets)
            self.portfolio_weights = np.zeros(self.n_assets)
            reward = 0.0
            self.done = True
        
        obs = self._get_observation()
        info = {
            "commission": total_commission, 
            "slippage": slippage,
            "cash": self.cash,
            "shares": self.shares,
            "portfolio_value": new_portfolio_value,
            "min_weight_applied": any(below_min) if np.sum(action) > 0 else False
        }
        return obs, reward, self.done, False, info
    
    def _get_observation(self):
        """Obtiene la observación actual del entorno."""
        try:
            # Prevenir NaN en la observación
            cash = max(self.cash, 0)  # Asegurar que no sea negativo
            prices = self.data[self.current_step]
            
            # Prevenir precios negativos o cero
            prices = np.maximum(prices, 1e-8)
            
            obs = np.concatenate(([cash], prices))
            
            # Verificar NaN en la observación
            if np.any(np.isnan(obs)):
                print("¡ADVERTENCIA! Observación con valores NaN generada:")
                print(f"Cash: {cash}, Prices: {prices}")
                # Retornar una observación segura
                return np.ones(self.n_assets + 1)
                
            return obs
            
        except Exception as e:
            print(f"Error en _get_observation(): {e}")
            # Retornar una observación segura como fallback
            return np.ones(self.n_assets + 1)

    def render(self):
        """Renderiza el estado actual del entorno."""
        try:
            asset_values = self.shares * self.data[self.current_step]
            total_asset_value = np.sum(asset_values)
            print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Cash: {self.cash:.2f}, Assets: {total_asset_value:.2f}")
            print(f"Shares: {self.shares}")
            print(f"Weights: {self.portfolio_weights}")
        except Exception as e:
            print(f"Error en render(): {e}")
    
    def _clip_values(self):
        """Limita los valores a rangos razonables para evitar inestabilidad numérica."""
        # Limitar acciones a valores no negativos
        self.shares = np.maximum(self.shares, 0)
        
        # Limitar balance mínimo
        if self.balance < 1.0:  # Si el balance cae por debajo de $1
            self.balance = 1.0  # Previene valores extremadamente pequeños o negativos
        
        # Limitar cash mínimo
        if self.cash < 0:
            self.cash = 0
        
        # Limitar pesos del portafolio
        self.portfolio_weights = np.clip(self.portfolio_weights, 0, 1)
        
        # Normalizar pesos si la suma no es cero
        sum_weights = np.sum(self.portfolio_weights)
        if sum_weights > 0:
            self.portfolio_weights = self.portfolio_weights / sum_weights