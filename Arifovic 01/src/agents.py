import mesa
from src.base.GA import EvaluableGene

"""
    EvaluableGene is to couple the agents with the genetic algorithm
"""
class GA_Agent(mesa.Agent):
    def __init__(self, unique_id:int, model:mesa.Model, evaluable_gene: EvaluableGene):
        super().__init__(unique_id, model)

        self.evaluable_gene: EvaluableGene = evaluable_gene

