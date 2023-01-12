from typing import Tuple
import mesa
from src.base.GA import Gene

class Gene20(Gene):
    def encode(self):
        return 20
    
    def breed(self, gene2: 'Gene') -> Tuple['Gene', 'Gene']:
        return (Gene20(self.string+"aaa"), Gene20(gene2.string + "bbb"))

    def mutate(self) -> 'Gene':
        return Gene20(self.string+"_changed")
    

def main():
    g1 = Gene20('first')
    g2 = Gene20('second')
    g3, g4 = g1*g2

    print(g3)


if __name__ == "__main__":
    main()