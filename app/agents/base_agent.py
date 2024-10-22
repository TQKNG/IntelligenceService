from abc import ABC, abstractmethod

class BaseAgent(ABC):

    def __init__(self, name, config: dict):
        self.name = name
        self.config = config
            
    @abstractmethod
    def perform_task(self, input_data):
        pass