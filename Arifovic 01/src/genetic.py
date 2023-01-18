import random
from typing import Callable, Tuple, Union, List

from src.base.GA import BaseGene, BaseGeneticAlgorithm, EvaluableGene
import numpy as np

class AGene(BaseGene):

    N:int = 30
    CONSUMPTION_SEG:int = 20
    LAMBDA_SEG: int = 10

    def __init__(self, string : np.ndarray = None):
        self.string : np.ndarray
        super().__init__(string)

    @classmethod
    def generate_gene(cls):
        return (np.random.rand(cls.N) >= 0.5)
    
    @classmethod
    def validate(cls, str:np.ndarray) :
        if not isinstance(str, np.ndarray):
            return False
        
        if not str.shape == (cls.N, ):
            return False
        
        if str.dtype is not np.dtype('bool'):
            return False
        
        return True

    def encode(self):
        ## first consumption_seg 
        return self.consumption_1(), self.portfolio() 
    
    def consumption_1(self):
        code = self.string[:self.CONSUMPTION_SEG]
        return code.dot(1 << np.arange(self.CONSUMPTION_SEG))

    def portfolio(self):
        code = self.string[self.CONSUMPTION_SEG: ]
        assert code.size == self.LAMBDA_SEG
        return code.dot(1 << np.arange(self.LAMBDA_SEG))

    def breed(self, gene2: 'AGene') -> Tuple['AGene', 'AGene']:
        random_position = random.randint(0, self.N - 1)

        string_11 = self.string[:random_position]
        string_12 = self.string[random_position:]

        string_21 = gene2.string[:random_position]
        string_22 = gene2.string[random_position:]

        offspring_string_1 = np.concatenate([string_11, string_22])
        offspring_string_2 = np.concatenate([string_21, string_12])
        return AGene(offspring_string_1), AGene(offspring_string_2)
    
    def mutate(self) -> 'AGene':
        random_position = random.randint(0, self.N)
        new_string = self.string.copy()
        new_string[random_position] = not new_string[random_position]

        return AGene(new_string)


ValidationFunction = Callable[[BaseGene], bool]

class AGeneticAlgorithm(BaseGeneticAlgorithm):
    def __init__(self, p_cross:float = 0.6, p_mut:float = 0.033):
        super().__init__()
        self.p_cross = p_cross
        self.p_mut = p_mut
    
    def register_validation_fn(self, validation_fn: ValidationFunction):
        self.validate_gene_fn = validation_fn
        return self 
        
    def add_agent(self, 
                    agent:EvaluableGene, 
                    custom_validation_fn: ValidationFunction = None)->None: 

        ## In case there is a custom validation function for the agent.
        validation_fn : ValidationFunction = self.validate_gene_fn
        if custom_validation_fn is not None:
            validation_fn = custom_validation_fn

        if not validation_fn(agent.gene.string):
            raise ValueError("Could not validate the agent's gene")
        
        ## only append if successfully validated
        self.agents.append(agent)
            

    def register_agents(self, new_agents: Union[EvaluableGene, List[EvaluableGene]]) -> None:
        if type(new_agents) is list:
            [self.add_agent(agent) for agent in new_agents]
            return 

        self.add_agent(new_agents)
        return 

    def remove_agent(self, agent:EvaluableGene):
        ## check if in list
        try:
            self.agents.remove(agent)
        except:
            raise ValueError("Removing agent not in pool")
        
    def remove_agents(self, agents: Union[EvaluableGene, List[EvaluableGene]]):
        if type(agents) is list:
            [self.remove_agent(agent) for agent in agents]
            return
        self.remove_agent(agents) 
    
    def reproduction_stage(self) -> None:
        N_TOURNAMENT = len(self.agents)

        self.winner_agents : List[EvaluableGene] = []
        for _ in range(N_TOURNAMENT):
            pairs_to_compare = random.sample(self.agents, 2)
            self.winner_agents.append(
                max(pairs_to_compare, key=lambda a : a.fitness)
            )
        return

    def crossover_stage(self):
        random.shuffle(self.winner_agents)
        if not len(self.winner_agents) % 2 == 0:
            self.winner_agents.pop()
        
        N_PARENTS = int (len(self.winner_agents) / 2 )

        moms = self.winner_agents[:N_PARENTS]
        dads = self.winner_agents[N_PARENTS:]
        assert len(moms) == len(dads)
        del self.winner_agents

        self.families:List[EvaluableGene] = []
        
        for mom, dad in zip(moms, dads):
            offspring_gene_1, offspring_gene_2 = mom.gene * dad.gene
            self.families.append(
                [
                    mom, 
                    dad, 
                    EvaluableGene(offspring_gene_1),
                    EvaluableGene(offspring_gene_2)
                ]
            )
        return 

    def mutation_stage(self):
        return super().mutation_stage()

    def election_stage(self):
        return super().election_stage()
    
