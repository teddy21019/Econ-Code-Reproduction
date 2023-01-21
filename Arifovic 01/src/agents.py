from typing import Callable, Dict
import mesa
from src.base.GA import EvaluableGene
from src.model import CurrencySubstitutionModel

"""
    EvaluableGene is to couple the agents with the genetic algorithm
"""
class GA_Agent(mesa.Agent):
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
        self.consumption_1, self.portfolio_1 = self.evaluable_gene.gene.encode() 

        self.saving = self.endowment_1 - self.consumption_1
        self.currency_1_holding = self.portfolio_1 * self.saving
        self.currency_2_holding = (1 - self.portfolio_1) * self.saving

    def old_step(self):
        evaluation_function = self.model.gene_evaluation_fn()

        self.evaluable_gene.fitness = evaluation_function(
            self.evaluable_gene.gene
        )
