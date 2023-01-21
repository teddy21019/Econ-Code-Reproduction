from src.genetic import AGene, AGeneticAlgorithm
import numpy as np

from src.model import CurrencySubstitutionModel
from mesa.time import RandomActivation

def main():
    ga = AGeneticAlgorithm(p_cross=0.6, p_mut=0.33)

    abm_schedular = RandomActivation
    
    model = CurrencySubstitutionModel(
                genetic_algo_class  =   ga, 
                n_agents            =   30,
                G_1                 =   10,
                G_2                 =   1,
                scheduler_constructor =   abm_schedular)
    
    model.random.shuffle

if __name__ == "__main__":
    main()