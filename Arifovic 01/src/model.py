from typing import Callable
import mesa 
from dataclasses import dataclass
from src.base.GA import GeneticAlgorithm, Gene

@dataclass
class CurrencySubstitutionModel(mesa.Model):
    def __init__(self, 
                genetic_algo_class : GeneticAlgorithm,
                n_agents: int   = 30,
                G_1     : float = 10,
                G_2     : float = 1,
                ):
        pass

    def gene_evaluation_fn(self) ->  Callable[..., float]:
        def ev_fn(gene:Gene) -> float:
            fitness = None
            return fitness
        return ev_fn