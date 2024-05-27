import pickle
import sys
from pathlib import Path


def main():
    path_to_results = Path(sys.argv[1])
    with open(path_to_results, mode='rb') as file:
        results = pickle.load(file=file)
    print(results)


if __name__ == '__main__':
    main()
