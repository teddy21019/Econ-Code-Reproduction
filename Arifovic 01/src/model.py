from collections import deque
from operator import ge
from typing import Callable, Deque, Dict, Tuple
import mesa
from src.agents import GA_Agent
from src.base.GA import BaseGeneticAlgorithm, BaseGene, EvaluableGene
from src.genetic import AGene, AGeneticAlgorithm

SchedulerConstructor =  Callable[[mesa.Model], mesa.time.BaseScheduler]

class CurrencySubstitutionModel(mesa.Model):
    def __init__(self, 
                genetic_algo_class : BaseGeneticAlgorithm,
                scheduler_constructor: SchedulerConstructor,
                n_agents: int   = 30,
                G_1     : float = 10,
                G_2     : float = 1,
                generation_num : int  = 2
                ):
        self.ga         : BaseGeneticAlgorithm     = genetic_algo_class
        self.n_agents   : int                   = n_agents      ## agents per generation
        self.G_1        : float                 = G_1
        self.G_2        : float                 = G_2
        self.scheduler_constructor  : SchedulerConstructor = scheduler_constructor

        ## initialize currenct price
        self.p1         : float = 1
        self.p2         : float = 1

        self.Lp1        : float = 1
        self.Lp2        : float = 1

        self.generation_list : Deque[mesa.time.BaseScheduler] = deque(maxlen=generation_num)
        self._max_generation_num = generation_num

        self.generate_agents()
        self.unique_id_generator = (i for i in range(1e5))


    def gene_evaluation_fn(self) ->  Callable[[BaseGene], float]:
        def ev_fn(gene:BaseGene) -> float:
            fitness = 0
            return fitness
        return ev_fn
    
    
    def generate_agents(self):
        """
            Generated the first generation data.
            Fake generation is constructed for convenient.
        """
        
        for gen_num in range(self._max_generation_num):

            ## create a scheduler for this gen

            gen_scheduler : mesa.time.BaseScheduler = self.scheduler_constructor(self) 

            ## add agents into scheduler
            for _ in range(self.n_agents):
                new_id = next(self.unique_id_generator)
                evaluable_gene = EvaluableGene(AGene())

                a = GA_Agent(
                    unique_id=new_id, 
                    model=self, 
                    evaluable_gene=evaluable_gene,  ## dependency injection of gene+value object
                    endowment_1=10,
                    endowment_2=1,
                    gen=gen_num) 

                gen_scheduler.add(a)
            
            # add the scheduler for this generation into the generation_list deque

            self.generation_list.appendleft(gen_scheduler)

    def step(self):
        """
        ## Step function for Arifovic 2001.

            Detail for agent is encapsulated in Agent class

            After each agent decides their consumption, portfolio, 
            we can then calculate the price of the two currency

            After the calculation of currency in first period, 
            the fitness value for the previous generation can then be 
        """

        ## young ones move first 

        self.generation_list[0].step()

        self.calculate_prices()

        ## then the old moves and calculate their consumption
        self.generation_list[1].step()

        ## at this time, the old agent should already evaluate its fitness 



    def calculate_prices(self) -> Tuple[float, float]:
        """
            Calculates the currency price for 1 and 2 
        """
        pass
    
    
