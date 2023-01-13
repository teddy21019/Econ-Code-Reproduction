from src.genetic import AGene
import numpy as np


def main():

    print(np.array([False, False, False, True]).dot(1 << np.arange(4)))

if __name__ == "__main__":
    main()