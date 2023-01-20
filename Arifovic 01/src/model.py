from typing import Callable
import mesa 
from dataclasses import dataclass
from src.base.GA import BaseGeneticAlgorithm, BaseGene

@dataclass
class CurrencySubstitutionModel(mesa.Model):
    def __init__(self, 
                genetic_algo_class : BaseGeneticAlgorithm,
                n_agents: int   = 30,
                G_1     : float = 10,
                G_2     : float = 1,
                ):
        self.ga         : AGeneticAlgorithm     = genetic_algo_class
        self.n_agents   : int                   = n_agents      ## agents per generation
        self.G_1        : float                 = G_1
        self.G_2        : float                 = G_2
        self.scheduler  : OGActivation = scheduler_constructor(self)

    def gene_evaluation_fn(self) ->  Callable[[BaseGene], float]:
        def ev_fn(gene:BaseGene) -> float:
            fitness = None
            return fitness
        return ev_fn
    
    
    def generate_agents(self):
        """
            Generated the first generation data
        """
        
        for _ in self.n_agents:
            new_id = next(self.unique_id_generator)
            evaluable_gene = EvaluableGene(AGene())

            a = GA_Agent(
                unique_id=new_id, 
                model=self, 
                evaluable_gene=evaluable_gene)  ## dependency injection of gene+value object
            self.scheduler.add(a)
        ## In this step, both scheduler are initialized with young people. 
    

    def step(self):
        """
        ## Step function for Arifovic 2001.

            Detail for agent is encapsulated in Agent class

            After each agent decides their consumption, portfolio, 
            we can then calculate the price of the two currency

            After the calculation of currency in first period, 
            the fitness value for the previous generation can then be 
        """

        ## Agents move with random order
        self.scheduler.step()
        

    def calculate_prices(self):
        pass
    
    
