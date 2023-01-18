from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Union
import numpy as np 
from abc import ABC, abstractmethod, abstractclassmethod

class BaseGene(ABC):
    """
        Gene is used to store the genetic characteristic of an agent.
    """
    def __init__(self, string = None):
        if string is None:
            self.string = self.generate_gene()  
            return 
        
        if not self.validate(string):
            raise TypeError(f"string type {type(string)} not allowed.")
        
        self.string = string

        return


    @abstractclassmethod
    def generate_gene(cls):
        ...
    
    @abstractclassmethod
    def validate(cls, string) -> bool:
        ...

    @abstractmethod
    def encode(self):
        """
            The return type depends on the particular structure considered.
        """
        ...


    @abstractmethod
    def breed(self, gene2:'BaseGene')->Tuple['BaseGene', 'BaseGene']:
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
    def mutate(self) -> 'BaseGene':
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
    def __mul__(self, gene_2) -> Tuple['BaseGene', 'BaseGene']:
        """Syntax candy for breeding"""
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
    gene : BaseGene
    fitness : float = field(default=None, compare=True) # compare=True so that the > < can be used on comparing performance



class BaseGeneticAlgorithm(ABC):
    """
        The GA class takes a list of agent (in particular an `EvaluableGene` interface objects)
        The agents must have a gene that performs breed, mutate, encode, ... etc. 
    """
    def __init__(self):
        self.agents : List[EvaluableGene] = []

    def set_evaluation_function(self, fn: Callable[..., float]):
        self.set_evaluation_function = fn
        return self


    @abstractmethod
    def register_agents(self, agents:Union[EvaluableGene, List[EvaluableGene]]):    ## change to iterable interface?
        ... 

    @abstractmethod
    def remove_agents(self, agents:Union[EvaluableGene, List[EvaluableGene]]):
        ...
    
    @abstractmethod
    def reproduction_stage(self):
        """ Choose (with repetition) genes with better fitness value.
            The GA object is now an mating pool.
        """
        ...

    @abstractmethod
    def crossover_stage(self):
        """ From the mating pool, random pair and make them breed, get offsprings"""
        ...

    @abstractmethod
    def mutation_stage(self):
        """ Mutation on offsprings"""
        ...

    @abstractmethod
    def election_stage(self):
        """
            For each of the offspring, evaluate the potential fitness according to a callback function
            passed from the source ( the main model ). 

            In this way the GA algorithm is decoupled from the main model.
        """
        ...
        

