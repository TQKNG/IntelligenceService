from .base_agent import BaseAgent

class SupervisorAgent(BaseAgent):
    def __init__(self, name, config: dict):
        super().__init__(name, config)
    
    def perform_task(self, input_data):
        print("SupervisorPerforming task")