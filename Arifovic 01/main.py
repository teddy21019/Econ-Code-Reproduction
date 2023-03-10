from src.genetic import AGeneticAlgorithm

from src.model import CurrencySubstitutionModel
from mesa.time import RandomActivation

def main():
    ga = AGeneticAlgorithm(p_cross=0.6, p_mut=0.33)

    abm_schedular = RandomActivation
    
    model = CurrencySubstitutionModel(
                genetic_algo_class  =   ga, 
                n_agents            =   30,
                G_1                 =   0,
                G_2                 =   9,
                scheduler_constructor =   abm_schedular)

    for _ in tqdm(range(100)):
        model.step()    

if __name__ == "__main__":
    main()