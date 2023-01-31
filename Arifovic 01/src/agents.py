from __future__ import annotations
import math
from typing import Callable, Dict, Tuple, TYPE_CHECKING
import mesa
import numpy as np
from src.base.GA import BaseGene, EvaluableGene
from src.factory import DecodeFunction

if TYPE_CHECKING:
    """ See https://stackoverflow.com/questions/39740632/python-type-hinting-without-cyclic-imports """
    from src.model import CurrencySubstitutionModel

"""
    EvaluableGene is to couple the agents with the genetic algorithm
"""
class GA_Agent(mesa.Agent):


    CONSUMPTION_SEG:int = 20
    LAMBDA_SEG: int = 10

    def __init__(self, unique_id:int, 
                model: CurrencySubstitutionModel, 
                evaluable_gene: EvaluableGene, 
                decoder: DecodeFunction,
                gen:int = 0,
                endowment_1: float = 10, 
                endowment_2: float = 1
                ):
        super().__init__(unique_id, model)

        """ Necessary of overriding the model declaration :
            Since for genetic algorithm, the evaluation function callback is necessary 
            to proceed on the GA fitness. 

            A better method : isolate another abstract model that implements the evaluation function method 
            instead of directly type hinting a CurrencySubstitution model to decrease coupling.
        """
        self.model: CurrencySubstitutionModel = model
        self.evaluable_gene: EvaluableGene = evaluable_gene
        self.decoder: DecodeFunction = decoder
        self.endowment_1 = endowment_1
        self.endowment_2 = endowment_2

        self.current_generation = gen

        self.consumption_1 = 0.1
        self.consumption_2 = 0.1
        self.saving = 0.1
        self.currency_1_holding = 0.1
        self.currency_2_holding = 0.1

        self.step_function_dict : Dict[int, Callable[..., None]] = {
            0 : self.young_step,
            1 : self.old_step
        }
    def step(self):

        if self.current_generation not in self.step_function_dict:
            raise ValueError("Too old! Check main model and make sure the agents don't live till they are too old to know what to do.")
        self.step_function_dict[self.current_generation]()

        self.current_generation += 1

    def step_through(self)->None:
        """ This is called only when a potential gene is needed to evaluate its fitness under the current economy """
        self.young_step()
        self.old_step()

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
        
        return_currency_1, return_currency_2 = self.model.currency_return ## need implementation

        # use up all the money to consume
        self.consumption_2 = self.endowment_2 + self.currency_1_holding * return_currency_1 + \
                                self.currency_2_holding * return_currency_2
        

        self.evaluable_gene.fitness =  self._evaluate()

    @property
    def fitness(self) -> float:
        return self.evaluable_gene.fitness

    def _evaluate(self) -> float:
        try:
            return math.log(self.consumption_1) + math.log(self.consumption_2)
        except AttributeError:
            raise AttributeError("Consumptions not found. Consumptions for each generation must be evaluated properly.") 
        except ValueError as ve:
            raise ValueError('Consumption cannot take log.')

    def _decode(self) -> Tuple[float,float]:
        gene = self.evaluable_gene.gene

        return self._decode_consumption_1(gene), self._decode_portfolio_1(gene)

    def _decode_consumption_1(self, gene: BaseGene) -> float:
        """
        The first 20 position of the gene. The implementation uses an `Agene` type for the evaluable gene type
        so an `np.ndarray` object is expecter and performed. 

        Should not be called except from `self._decode()`
        """
        decoded, n = self.decoder(gene, 'consumption_1')
        return decoded / (2**n- 1) * self.endowment_1

    def _decode_portfolio_1(self, gene: BaseGene) -> float:
        """
        The last 10 position of the gene.
        """
        decoded, n = self.decoder(gene, 'portfolio_1')
        return decoded / (2**n - 1)