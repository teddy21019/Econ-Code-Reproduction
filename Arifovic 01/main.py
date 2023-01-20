from src.genetic import AGene, AGeneticAlgorithm

from src.model import CurrencySubstitutionModel

from src.scheduler import OGActivation

def main():

    abm_schedular = OGActivation
    
    model = CurrencySubstitutionModel(
                genetic_algo_class  =   ga, 
                n_agents            =   30,
                G_1                 =   10,
                G_2                 =   1,
                scheduler           =   abm_schedular)
    
    model.random.shuffle

if __name__ == "__main__":
    main()