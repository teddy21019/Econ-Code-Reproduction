from typing import Tuple
import mesa
from src.base.GA import Gene

class GeneTest(Gene):
    def encode(self):
        pass 

    def breed(self, gene2: 'Gene') -> Tuple['Gene', 'Gene']:
        string_1 = self.string.split('.')
        string_2 = gene2.string.split('.')

        return GeneTest(f"{string_1[0]}.{string_2[1]}"), GeneTest(f"{string_2[0]}.{string_1[1]}")
    
    def mutate(self) -> 'Gene':
        return GeneTest(self.string + "_mut")


def main():
    g1 = GeneTest("10001.01101")
    g2 = GeneTest("01101.11111")
    g3, g4 = g1*g2
    print(g3.mutate())

if __name__ == "__main__":
    main()