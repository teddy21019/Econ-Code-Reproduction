from typing import Callable, Dict
import mesa
from src.base.GA import EvaluableGene

"""
    EvaluableGene is to couple the agents with the genetic algorithm
"""
class GA_Agent(mesa.Agent):
    def __init__(self, unique_id:int, 
                model:mesa.Model, 
                evaluable_gene: EvaluableGene, 
                endowment_1: float = 10, 
                endowment_2: float = 1):
        super().__init__(unique_id, model)

        self.evaluable_gene: EvaluableGene = evaluable_gene
        self.endowment_1 = endowment_1
        self.endowment_2 = endowment_2

    def step(self, gen: int = 0):
        step_function_dict : Dict[int, Callable[..., None]] = {
            0 : self.young_step,
            1 : self.old_step
        }

        if gen not in step_function_dict:
            raise ValueError("Too old! Generation index doesn't match agent method. The agent is too old to know what to do!")
        step_function_dict[gen]()

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
        pass
