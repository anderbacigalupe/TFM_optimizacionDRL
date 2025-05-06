# TFM - Optimización de Carteras de Inversión con Aprendizaje por Refuerzo

Este proyecto forma parte del Trabajo de Fin de Máster en Ciencia de Datos. El objetivo principal es aplicar algoritmos de **Aprendizaje por Refuerzo Profundo (Deep Reinforcement Learning)**, concretamente **DQN** y **DDPG**, para optimizar una cartera de inversión compuesta por ETFs. Además, se utiliza el modelo de **Media-Varianza de Markowitz** como benchmark, aplicando una estrategia **walk-forward** para la optimización.

---

## Estructura del proyecto

TFM_optimizacionDRL/
│
├── agentes/                # Implementaciones de agentes DRL
├── data/                   # Datos históricos de precios y variables económicas (CSV)
├── entorno/                # Implementación del entorno de inversión personalizado
├── entrenamiento/          # Entrenamiento de los modelos DQN y DDPG
├── evaluacion/             # Scripts para calcular metricas de los modelos de DRL
├── markowitz/              # Implementación del modelo Media-Varianza con walk-forward
├── results/                # Resultados finales de pruebas y comparaciones
├── utils/                  # 
├── .gitignore              # Archivos y carpetas ignoradas por Git
├── environment.yml         # Archivo para reproducir el entorno con conda
├── README.md               # Este archivo
└── requirements.txt        


---

##  Algoritmos implementados

- ✅ Deep Q-Network (DQN)
- ✅ Deep Deterministic Policy Gradient (DDPG)
- ✅ Markowitz (Media-Varianza, benchmark)

---

##  Metodología

- **Preprocesamiento** de los datos financieros
- Definición de un **entorno personalizado** para el agente
- **Entrenamiento y evaluación** de los agentes DQN y DDPG
- Comparación de rendimiento frente al modelo de **Markowitz** con optimización walk-forward

---

## Requisitos

- Python 3.10+
- Entorno gestionado con `conda`
- Paquetes:
  - `gym`, `numpy`, `pandas`, `matplotlib`, `stable-baselines3`, `yfinance`, `scikit-learn`, etc.

> Los requisitos exactos están definidos en el archivo `environment.yml`.

---

## Cómo ejecutar

1. Clona el repositorio:
   ```bash
   git clone https://github.com/TU_USUARIO/TFM_optimizacionDRL.git
   cd TFM_optimizacionDRL
