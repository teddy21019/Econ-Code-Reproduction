import math
from typing import Callable, Dict, Tuple
import mesa
import numpy as np
from src.base.GA import EvaluableGene
from src.model import CurrencySubstitutionModel

"""
    EvaluableGene is to couple the agents with the genetic algorithm
"""
class GA_Agent(mesa.Agent):


    CONSUMPTION_SEG:int = 20
    LAMBDA_SEG: int = 10

    def __init__(self, unique_id:int, 
                model:CurrencySubstitutionModel, 
                evaluable_gene: EvaluableGene, 
                gen:int = 0,
                endowment_1: float = 10, 
                endowment_2: float = 1):
        super().__init__(unique_id, model)

        """ Necessary of overriding the model declaration :
            Since for genetic algorithm, the evaluation function callback is necessary 
            to proceed on the GA fitness. 

            A better method : isolate another abstract model that implements the evaluation function method 
            instead of directly type hinting a CurrencySubstitution model to decrease coupling.
        """
        self.model : CurrencySubstitutionModel  = model
        self.evaluable_gene: EvaluableGene = evaluable_gene
        self.endowment_1 = endowment_1
        self.endowment_2 = endowment_2

        self.current_generation = gen

        self.step_function_dict : Dict[int, Callable[..., None]] = {
            0 : self.young_step,
            1 : self.old_step
        }
    def step(self):

        if self.current_generation not in self.step_function_dict:
            raise ValueError("Too old! Check main model and make sure the agents don't live till they are too old to know what to do.")
        self.step_function_dict[self.current_generation]()

        self.current_generation += 1

    def young_step(self):
        """
            The young agent decodes its string and calculates its first period consumption 
            and portfolio decision
        """

        ## change to a dict in future
        self.consumption_1, self.portfolio_1 = self._decode()

        self.saving = self.endowment_1 - self.consumption_1
        self.currency_1_holding = self.portfolio_1 * self.saving
        self.currency_2_holding = (1 - self.portfolio_1) * self.saving

    def old_step(self):

        ## self.consuption_2 is first set here to prevent early access to the attribute. 
        ## This prevents an early call of evaluate function
        
        price_currency_1, price_currency_2 = self.model.currency_prices ## need implementation

        # use up all the money to consume
        self.consumption_2 = self.currency_1_holding / price_currency_1 + \
                                self.currency_2_holding / price_currency_2
        

        self.evaluable_gene.fitness =  self.evaluate()

    def evaluate(self) -> float:
        try:
            return math.log(self.consumption_1) + math.log(self.consumption_2)
        except AttributeError:
            raise AttributeError("Consumptions not found. Consumptions for each generation must be evaluated properly.") 
        except ValueError as ve:
            raise ValueError('Consumption cannot take log.')

    def _decode(self) -> Tuple[float,float]:
        gene_string: np.ndarray = self.evaluable_gene.gene.string

        return self._decode_consumption_1(gene_string), self._decode_portfolio_1(gene_string)

    def _decode_consumption_1(self, gene_string: np.ndarray) -> float:
        """
        The first 20 position of the gene. The implementation uses an `Agene` type for the evaluable gene type
        so an `np.ndarray` object is expecter and performed. 

        Should not be called except from `self._decode()`
        """
        code = gene_string[:self.CONSUMPTION_SEG]
        return code.dot(1 << np.arange(self.CONSUMPTION_SEG))

    def _decode_portfolio_1(self, gene_string:np.ndarray) -> float:
        """
        The last 10 position of the gene.
        """
        code = gene_string[self.CONSUMPTION_SEG: ]
        assert code.size == self.LAMBDA_SEG
        return code.dot(1 << np.arange(self.LAMBDA_SEG))