
class BasePrompt:
    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs