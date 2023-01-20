from typing import Callable
import mesa
from src.agents import GA_Agent, OldAgent, YoungAgent 
from src.base.GA import BaseGeneticAlgorithm, BaseGene, EvaluableGene
from src.genetic import AGene, AGeneticAlgorithm

class CurrencySubstitutionModel(mesa.Model):
    def __init__(self, 
                genetic_algo_class : BaseGeneticAlgorithm,
                scheduler_constructor: Callable[[mesa.Model], mesa.time.BaseScheduler],
                n_agents: int   = 30,
                G_1     : float = 10,
                G_2     : float = 1,
                ):
        self.ga         : AGeneticAlgorithm     = genetic_algo_class
        self.n_agents   : int                   = n_agents      ## agents per generation
        self.G_1        : float                 = G_1
        self.G_2        : float                 = G_2
        self.scheduler  : mesa.time.RandomActivationByType = scheduler_constructor(self)

        ## initialize currenct price
        self.p1 = 1
        self.p2 = 1

        self.Lp1 = 1
        self.Lp2 = 1

        self.generate_agents()
        self.unique_id_generator = (i for i in range(10e5))


    def gene_evaluation_fn(self) ->  Callable[..., float]:
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

            a = YoungAgent(
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

        ## Young agent decodes and calculate consumption/ portfolio
        self.scheduler.step_type(YoungAgent)   ## In this case is the random-activation-by-type class. 
        
        ## Evaluate the price of each currency
        self.calculate_prices()        

        self.scheduler.step_type(OldAgent)

    def calculate_prices(self):
        pass
    
    
