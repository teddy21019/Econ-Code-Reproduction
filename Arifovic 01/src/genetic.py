from base64 import encode
import random
from typing import Tuple

from mesa import Agent
import src.base.GA as GA 
import numpy as np

class AGene(GA.BaseGene):

    N:int = 30
    CONSUMPTION_SEG:int = 20
    LAMBDA_SEG: int = 10

    def __init__(self, string : np.ndarray = None):
        self.string : np.ndarray
        super().__init__(string)

    @classmethod
    def generate_gene(cls):
        return (np.random.rand(cls.N) >= 0.5)
    
    @classmethod
    def validate(cls, str:np.ndarray) :
        if not isinstance(str, np.ndarray):
            return False
        
        if not str.shape == (cls.N, ):
            return False
        
        if str.dtype is not np.dtype('bool'):
            return False
        
        return True

    def encode(self):
        ## first consumption_seg 
        return self.consumption_1(), self.portfolio() 
    
    def consumption_1(self):
        code = self.string[:self.CONSUMPTION_SEG]
        return code.dot(1 << np.arange(self.CONSUMPTION_SEG))

    def portfolio(self):
        code = self.string[self.CONSUMPTION_SEG: ]
        assert code.size == self.LAMBDA_SEG
        return code.dot(1 << np.arange(self.LAMBDA_SEG))

    def breed(self, gene2: 'AGene') -> Tuple['AGene', 'AGene']:
        random_position = random.randint(0, self.N - 1)

        string_11 = self.string[:random_position]
        string_12 = self.string[random_position:]

        string_21 = gene2.string[:random_position]
        string_22 = gene2.string[random_position:]

        offspring_string_1 = np.concatenate([string_11, string_22])
        offspring_string_2 = np.concatenate([string_21, string_12])
        return AGene(offspring_string_1), AGene(offspring_string_2)
    
    def mutate(self) -> 'AGene':
        random_position = random.randint(0, self.N)
        new_string = self.string.copy()
        new_string[random_position] = not new_string[random_position]

        return AGene(new_string)




class AGeneticAlgorithm(GA.BaseGeneticAlgorithm):
    pass 