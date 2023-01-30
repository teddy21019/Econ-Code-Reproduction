from collections import deque
from math import isnan
from typing import Callable, Deque, Dict, List, Tuple, Type
import mesa
from src.agents import GA_Agent
from src.base.GA import BaseGeneticAlgorithm, BaseGene, EvaluableGene
from src.genetic import AGene, AGeneticAlgorithm
import numpy as np
import pandas as pd


# SchedulerConstructor =  Callable[[mesa.Model], mesa.time.BaseScheduler]
SchedulerConstructor = Type[mesa.time.BaseScheduler]
class CurrencySubstitutionModel(mesa.Model):
    def __init__(self, 
                genetic_algo_class : BaseGeneticAlgorithm,
                scheduler_constructor: SchedulerConstructor,
                n_agents: int   = 30,
                G_1     : float = 0,
                G_2     : float = 10,
                endowment_1 : float = 10, 
                endowment_2 : float = 1,
                generation_num : int  = 2
                ):
        self.ga         : BaseGeneticAlgorithm     = genetic_algo_class
        self.n_agents   : int                   = n_agents      ## agents per generation
        self.G_1        : float                 = G_1
        self.G_2        : float                 = G_2
        self.endowment_1 : float                = endowment_1
        self.endowment_2 : float                = endowment_2
        self.scheduler_constructor  : SchedulerConstructor = scheduler_constructor

        ## initialize currenct price
        self._p1         : float = 1
        self._p2         : float = 1
        ## stores previous price
        self._Lp1        : float = 1
        self._Lp2        : float = 1

        ## Total money supply
        ## Initially set Hi_0 both at 10k
        self._H1         : float = 100
        self._H2         : float = 100
        ## Stores the previous money supply
        self._LH1        : float = 100
        self._LH2        : float = 100


        self.generation_list : Deque[mesa.time.BaseScheduler] = deque(maxlen=generation_num)
        self._max_generation_num = generation_num

        self.unique_id_generator = (i for i in range(100_000_00))
        self.time_generator = (t for t in range(100_000_00))

        self.generate_agents()

        self.datacollector : List[Dict[str, float]] = []


    def gene_evaluation_fn(self) ->  Callable[[BaseGene], float]:
        """
        A closure that returns a function that takes in a gene and outputs a fitness.

        The fitness is constructed by creating a fake agent that evolve under this economy,
        without affecting the economy.
        """
        def ev_fn(gene:BaseGene) -> float:
            
            potential_agent = self.new_agent(gene)

            potential_agent.step_through()
            """ The evaluable gene object should have the fitness by now"""

            return potential_agent.fitness
        return ev_fn
    
    
    def generate_agents(self):
        """
            Generated the first generation data.
            Fake generation is constructed for convenient.
        """
        
        for gen_num in reversed(range(self._max_generation_num)):

            ## create a scheduler for this gen

            gen_scheduler : mesa.time.BaseScheduler = self.scheduler_constructor(self) 

            ## add agents into scheduler
            for _ in range(self.n_agents):

                a = self.new_agent(AGene(), gen_num)
                a.step_through()            ## step through to obtain initial value of consumption
                gen_scheduler.add(a)
            
            # add the scheduler for this generation into the generation_list deque

            self.generation_list.appendleft(gen_scheduler)

    def new_agent(self, gene:BaseGene, gen_num:int = 0):
        new_id = next(self.unique_id_generator)
        evaluable_gene = EvaluableGene(gene)

        return GA_Agent(
            unique_id=new_id,
            model=self,
            evaluable_gene=evaluable_gene,  ## dependency injection of gene+value object
            endowment_1=self.endowment_1,
            endowment_2=self.endowment_2,
            gen=gen_num)

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

        ## collect data before evolve
        self.collect_data()


        ## at this time, the old agent should already evaluate its fitness 

        gene_pool: List[EvaluableGene]= self.evolve()

        self.new_generation(gene_pool=gene_pool)



    def calculate_prices(self) -> Tuple[float, float]:   # type: ignore        
        """
            Calculates the currency price for 1 and 2 
        """
        ## store last price
        self._Lp1 = self._p1
        self._Lp2 = self._p2

        ## calculate price
        sum_of_agent_c1_saving = np.sum([ya.currency_1_holding for ya in self.youngs])
        sum_of_agent_c2_saving = np.sum([ya.currency_2_holding for ya in self.youngs])
        
        self._p1 = self._LH1 / (sum_of_agent_c1_saving - self.G_1)
        self._p2 = self._LH2 / (sum_of_agent_c2_saving - self.G_2)

        ## store last supply
        self._LH1 = self._H1
        self._LH2 = self._H2

        ## calculate next supply
        self._H1 = self._p1 * self.G_1 + self._LH1
        self._H2 = self._p2 * self.G_2 + self._LH2


    def evolve(self) -> List[EvaluableGene]:

        self.ga.clear()
        self.ga.set_evaluation_function(self.gene_evaluation_fn())
        self.ga.register_agents(
            [agent.evaluable_gene for agent in self.elders]
        )
        return self.ga.evolve()


    def new_generation(self, gene_pool:List[EvaluableGene]):
        
        ## shift to right, the last element( old ones ) now becomes the first
        self.generation_list.rotate(1)

        ## new scheduler
        new_gen_scheduler = self.scheduler_constructor(self)

        for eg in gene_pool:
            a = self.new_agent(eg.gene)
            new_gen_scheduler.add(a)

        self.generation_list[0] = new_gen_scheduler


    @property
    def elders(self) -> List[GA_Agent]:
        return self.generation_list[-1].agents # type: ignore

    @property
    def youngs(self) -> List[GA_Agent]:
        return self.generation_list[0].agents # type: ignore

    @property
    def currency_prices(self)->Tuple[float, float]:
        return self._p1, self._p2
    
    @property
    def currency_return(self) -> Tuple[float, float]:
        """
        ## Breaking of the double currency system

        During the aggregate step, the currency price might become negative infinity, 
        causing nan to apprear during runtime. 

        ## Solution
        Whenever there is negative "return" of a nan appearing in the price, 
        make it 0, indicating that it is simply valueless for the agents.
        """
        return_1, return_2 = 0.0 , 0.0
        if not (isnan(self._p1) or isnan(self._Lp1)):
            return_1 = max(self._Lp1 / self._p1, 0)

        
        if not (isnan(self._p2) or isnan(self._Lp2)):
            return_2 = max(self._Lp2 / self._p2, 0)

        
        return return_1, return_2
    

    def collect_data(self):
        new_row = {
                'T': next(self.time_generator) ,
                'Exchange Rate' : self._p1/self._p2,
                'Inflation Currency 1' : self._Lp1/self._p1,
                'Avg. Consumption 1 ': np.mean([ya.consumption_1 for ya in self.youngs]),
                'Avg. Portfolio 1'  :  np.mean([ya.portfolio_1 for ya in self.youngs]),
                'Avg. fitness'      :  np.mean([ea.fitness for ea in self.elders])
            }
        self.datacollector.append(new_row)