import mesa
from src.base.GA import EvaluableGene

"""
    EvaluableGene is to couple the agents with the genetic algorithm
"""
class GA_Agent(mesa.Agent):
    def __init__(self, unique_id:int, model:mesa.Model, evaluable_gene: EvaluableGene):
        super().__init__(unique_id, model)

        self.evaluable_gene: EvaluableGene = evaluable_gene

class YoungAgent(GA_Agent):
    pass

class OldAgent(GA_Agent):
    pass

def get_old(young: YoungAgent):
    return OldAgent(
        unique_id=young.unique_id,
        model = young.model,
        evaluable_gene=young.evaluable_gene
    )
