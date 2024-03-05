from implementations.pytorch.toolbox.datasets.builders.numeric_dataset_builder import PytorchNumericDatasetBuilder

PATH = "C:/Users/Joel/Desktop/dataset.csv"

for batch in PytorchNumericDatasetBuilder(PATH, 2, 42).build(0.6, 0.2, 0.2).test().batches():
    print(batch.inputs())
    print(batch.targets())
