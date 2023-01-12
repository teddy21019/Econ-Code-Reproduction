from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Union
import numpy as np 
from abc import ABC, abstractmethod

@dataclass
class Gene(ABC):
    """
        Gene is used to store the genetic characteristic of an agent.
    """
    string: np.array

    @abstractmethod
    def encode(self):
        """
            The return type depends on the particular structure considered.
        """
        ...


    @abstractmethod
    def breed(self, gene2:'Gene')->Tuple['Gene', 'Gene']:
        """
            Introduction
            ===========
            In genetic algorithm, a process called crossover is conducted. 
            During crossover, part of the characteristics will be swapped upon both genes.
            To prevent confusion, the specific method is called "breed".

            Parameter
            =========
            gene2 :  Gene
                --- the other gene to breed with

            Return
            =======
            A tuple of new genes (Gene, Gene)
        """
        ...
    
    @abstractmethod
    def mutate(self) -> 'Gene':
        """
        Introduction
        ======
            Mutate is another method for genetic algorithms. 
            For some segment or position of the string, an mutation is imposed
        Return
        ==========
        A Gene object, different to the current one
        """
        ...
    def __mul__(self, gene_2) -> Tuple['Gene', 'Gene']:
        return self.breed(gene_2)

@dataclass(order=True)
class EvaluableGene:
    """
        Class for agent-based modeling or other application.

        For cases that has to evaluate the performance of a gene, 
        this inheritance is needed. 

        It could be instantiated, as the main model can directly evaluate the performance of this gene. 
        It could also be inherited by an agent, while the code for evaluation in the main model remains unchange.

        This is the dependency inversion principle. 
    """
    gene : Gene
    fitness : float = field(default=None, compare=True) # compare=True so that the > < can be used on comparing performance


class GeneticAlgorithm(ABC):
    """
        The GA class takes a list of agent (in particular an `EvaluableGene` interface objects)
        The agents must have a gene that performs breed, mutate, encode, ... etc. 
    """
    def __init__(self, agents: List[EvaluableGene]):
        self.agents = agents


    @abstractmethod
    def register_agents(self, agents:Union[EvaluableGene, List[EvaluableGene]]):
        ... 

    @abstractmethod
    def remove_agents(self, agents:Union[EvaluableGene, List[EvaluableGene]]):
        ...
    
    @abstractmethod
    def reproduction_stage(self):
        ...

    @abstractmethod
    def crossover_stage(self):
        ...

    @abstractmethod
    def mutation_stage(self):
        ...

    @abstractmethod
    def election_stage(self, evaluation_fn: Callable[..., float]):
        """
            For each of the offspring, evaluate the potential fitness according to a callback function
            passed from the source ( the main model ). 

            In this way the GA algorithm is decoupled from the main model.
        """
        ...
        

