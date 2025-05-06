# TFM - Optimización de Carteras de Inversión con Aprendizaje por Refuerzo

Este proyecto forma parte del Trabajo de Fin de Máster en Ciencia de Datos. El objetivo principal es aplicar algoritmos de **Aprendizaje por Refuerzo Profundo (Deep Reinforcement Learning)**, concretamente **DQN** y **DDPG**, para optimizar una cartera de inversión compuesta por sesi **ETFs**. Además, se utiliza el modelo de **Media-Varianza de Markowitz** y una cartera equiponderada como benchmark, aplicando una estrategia walk-forward para la optimización

---

## Estructura del proyecto

TFM_optimizacionDRL/
│
├── agentes/                # Implementaciones de agentes DRL
│   ├── agente_dqn.py       # Implementación del agente DQN
│   └── agente_ddpg.py      # Implementación del agente DDPG (Actor-Crítico)
│
├── data/                   # Datos históricos de precios y variables económicas
│   ├── raw/                # Archivos CSV originales descargados de Yahoo Finance
│   └── processed/          # Datos procesados y combinados para entrenamiento
│
├── entorno/                # Implementación del entorno de inversión personalizado
│   └── entorno_cartera.py  # Entorno compatible con Gymnasium
│
├── entrenamiento/          # Scripts de entrenamiento de los modelos
│   ├── entrenar_dqn.py     # Entrenamiento del modelo DQN
│   └── entrenar_ddpg.py    # Entrenamiento del modelo DDPG
│
├── evaluacion/             # Evaluación y métricas de los modelos
│   ├── evaluar_dqn.py      # Evaluación del modelo DQN
│   └── evaluar_ddpg.py     # Evaluación del modelo DDPG
│
├── markowitz/              # Implementación del modelo Media-Varianza
│   ├── markowitz_backtest.py         # Backtesting del modelo de Markowitz
│   ├── simulacion_equiponderada.py   # Simulación de cartera equiponderada
│   └── simulacion_markowitz.py       # Simulación del modelo de Markowitz
│
├── modelos/                # Modelos entrenados (generado automáticamente)
│
├── results/             # Resultados y visualizaciones
│
├── utils/                  # Utilidades para procesamiento de datos
│   ├── procesar_datos.py           # Preprocesamiento de datos
│   └── calculate_log_returns.py    # Cálculo de retornos logarítmicos
│
├── .gitignore              # Archivos y carpetas ignoradas por Git
├── environment.yml         # Archivo para reproducir el entorno con conda
├── README.md               # Este archivo
└── requirements.txt        # Dependencias del proyecto        


---

##  Algoritmos implementados

- ✅ Deep Q-Network (DQN): Algoritmo de aprendizaje por refuerzo que discretiza el espacio de acciones para la asignación de pesos en la cartera.
- ✅ Deep Deterministic Policy Gradient (DDPG): Arquitectura actor-crítico que opera directamente en un espacio de acciones continuo, permitiendo una asignación de pesos más precisa.
- ✅ Markowitz (Media-Varianza): Implementación del modelo clásico de optimización con estrategia walk-forward anclada, usado como benchmark.

---

## Características principales

- Entorno personalizado basado en Gymnasium que simula operaciones en mercados financieros con costes de transacción realistas.
- Restricciones de diversificación que garantizan un peso mínimo del 5% por activo.
- Métricas financieras completas para evaluación: rendimiento anualizado, volatilidad, ratios de Sharpe y Sortino, VaR95 diario y máximo drawdown.
- Métodos de exploración adaptados: Epsilon-greedy para DQN y ruido Ornstein-Uhlenbeck para DDPG.

## ETFs utilizados

SPY: Renta variable (acciones estadounidenses)
GLD: Materia prima (oro, activo refugio)
TLT: Renta fija (bonos del Tesoro de EE.UU. a largo plazo)
HYG: Bonos de alto rendimiento (high yield)
VNQ: Sector inmobiliario
VB: Small caps de EEUU

---

## Requisitos

- Python 3.10+
- Dependencias (instalables mediante pip install -r requirements.txt):
    numpy>=1.20.0
    pandas>=1.3.0
    matplotlib>=3.4.0
    gymnasium>=0.28.1
    torch>=1.10.0
    yfinance>=0.1.74
    scikit-learn>=1.0.0
    tqdm>=4.62.0
    tensorboard>=2.10.0
    seaborn>=0.11.0
    ipython>=7.0.0
    json5>=0.9.6
    scipy>=1.7.0
    statsmodels>=0.13.0

---

## Cómo ejecutar

**1. Clonar el repositorio:**
git clone https://github.com/anderbacigalupe/TFM_optimizacionDRL.git
cd TFM_optimizacionDRL

**2. Instalar dependencias:**
pip install -r requirements.txt

**3. Descargar y procesar datos (si no están ya disponibles):**
python -m utils.procesar_datos
python -m utils.calculate_log_returns

**4. Entrenar modelos:**
# Entrenar DQN
python -m entrenamiento.entrenar_dqn

# Entrenar DDPG
python -m entrenamiento.entrenar_ddpg

**5. Evaluar modelos:**
# Evaluar DQN
python -m evaluacion.evaluar_dqn

# Evaluar DDPG
python -m evaluacion.evaluar_ddpg

**6. Ejecutar benchmark de Markowitz:**
python -m markowitz.simulacion_markowitz
python -m markowitz.simulacion_equiponderada
