from typing import Callable
import mesa
from src.agents import GA_Agent 
from src.base.GA import BaseGeneticAlgorithm, BaseGene, EvaluableGene
from src.genetic import AGene, AGeneticAlgorithm

class CurrencySubstitutionModel(mesa.Model):
    def __init__(self, 
                genetic_algo_class : BaseGeneticAlgorithm,
                scheduler: mesa.time.BaseScheduler,
                n_agents: int   = 30,
                G_1     : float = 10,
                G_2     : float = 1,
                ):
        self.ga         : AGeneticAlgorithm     = genetic_algo_class
        self.n_agents   : int                   = n_agents
        self.G_1        : float                 = G_1
        self.G_2        : float                 = G_2
        self.scheduler  : mesa.time.BaseScheduler = scheduler

        self.generate_agents()
        self.unique_id_generator = (i for i in range(10e5))


    def gene_evaluation_fn(self) ->  Callable[..., float]:
        def ev_fn(gene:BaseGene) -> float:
            fitness = None
            return fitness
        return ev_fn
    
    def generate_agents(self):
        for _ in self.n_agents:
            new_id = next(self.unique_id_generator)
            evaluable_gene = EvaluableGene(AGene())
            
            a = GA_Agent(
                unique_id=new_id, 
                model=self, 
                evaluable_gene=evaluable_gene)
            self.scheduler.add(a)

    def step(self):
        self.scheduler.step()