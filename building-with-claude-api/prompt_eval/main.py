import json
import sys
from pathlib import Path

from generate_test_dataset import generate_dataset
from model_grader import run_eval

dataset_file = Path("dataset.json")

def main():

    # test the dataset generation 

    dataset = generate_dataset()
    print(dataset)

    # saving the dataset
    with open('dataset.json', 'w') as f:
        json.dump(dataset, f, indent=2)

    if dataset_file.exists():
        with dataset_file.open("r") as f:
            dataset = json.load(f)
    else:
        dataset = generate_dataset()

        with dataset_file.open("w") as f:
            json.dump(dataset,f,indent=2)
            
    results = run_eval(dataset)

    print(json.dumps(results, indent=2))


if __name__== "__main__":
    sys.exit(main())