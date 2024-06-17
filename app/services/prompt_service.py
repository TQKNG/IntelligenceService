class PromptService:
    def __init__(self, prompt):
        self.prompt = prompt

    def get_prompt(self):
        return self.prompt

    def set_prompt(self, prompt):
        self.prompt = prompt