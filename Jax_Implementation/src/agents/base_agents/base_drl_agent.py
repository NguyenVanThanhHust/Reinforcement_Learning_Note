from abc import ABC, abstractclassmethod

class BaseDeepRLAgent(ABC):
    def __init__(self) -> None:
        super().__init__()