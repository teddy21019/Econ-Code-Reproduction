import mesa
from src.base.GA import EvaluableGene

"""
    EvaluableGene is to couple the agents with the genetic algorithm
"""
class GA_Agent(mesa.agents, EvaluableGene):
    pass

