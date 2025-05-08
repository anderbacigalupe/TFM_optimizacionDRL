# test_agent.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from agentes import DQNAgent
import inspect

print(f"DQNAgent parameters: {inspect.signature(DQNAgent.__init__)}")
print(f"DQNAgent source file: {inspect.getfile(DQNAgent)}")