# test_agent.py
from agentes.agente_dqn import DQNAgent
import inspect

print(f"DQNAgent parameters: {inspect.signature(DQNAgent.__init__)}")
print(f"DQNAgent source file: {inspect.getfile(DQNAgent)}")