# Framework for the design and lifecycle of neural networks
[![Skill Icons](https://skillicons.dev/icons?i=py,pytorch&perline=10)](https://skillicons.dev)

## Introduction

This framework streamlines the design, training, evaluation, and deployment of neural networks isolating developers from the technical details of deep learning libraries. Its goal is to allow developers to focus on the logic and structure of the networks to be developed, thus reducing complexity and promoting efficiency and collaboration between teams with different expertise. It offers a comprehensive suite of tools to simplify each stage of the neural network lifecycle, including architecture customization, hyperparameter tuning, model validation, and deployment. With an intuitive interface and robust automation features, it enhances productivity and accelerates the development of high-performance neural networks.


## Table of Contents
1. [Overview](#overview)
    - [Key Concepts](#key-concepts)
      - [Architecture](#architecture)
        - [Section](#section)
        - [Block](#block)
        - [Layer](#layer)
      - [Dataset](#dataset)
      - [Experiment](#experiment)
      - [Laboratory](#laboratory)
2. [Installation](#installation)
3. [Examples](#examples)
    - [Artificial Neural Network](#artificial-neural-network)
    - [Convolutional Neural Network](#convolutional-neural-network)
    - [Recurrent Neural Network](#recurrent-neural-network)
4. [License](#license)
5. [Contact](#contact)
## Overview

This framework focuses on two main aspects:
- **Design**: Configuring the architecture of the neural network, such as layers and activation functions.
- **Lifecycle Management**: Managing the lifecycle of the neural network, training and evaluation.

### Key Concepts

### Architecture
The Architecture defines the structure of the neural network. It specifies the types of layers, the number of units in each layer, and the activation functions. The architecture can be composed of either sections, blocks or layers. 

Example:
```
(Architecture("e626")
    .attach(recurrentSection([
                Block([
                    LSTMLayer(input_size=28, hidden_size=512, output_type=LSTMLayer.OutputType.EndSequence, num_layer=4, bidirectional=False, dropout=0.0),
                    FlattenLayer(from_dim=2, to_dim=1), 
                    ReLULayer()
                ]), 
                Block([
                    LinearLayer(in_features=14336, out_features=300, dimension=-1, bias=True), 
                    ReLULayer()
                ]), 
                Block([
                    LinearLayer(in_features=300, out_features=75, dimension=-1, bias=True), 
                    ReLULayer()
                ]), 
                Block([
                    LinearLayer(in_features=75, out_features=10, dimension=-1, bias=True), 
                    SoftmaxLayer(dimension=-1)
                ])
])))
```

#### Section
Sections are a set of blocks grouped together to perform a specific task. Linear sections as well as convolutional sections and recurrent sections are the possibilities from which to choose depending on the task. 

#### Block
Blocks are a set of layers that are commonly grouped together. Aside of the simple block, there is an implementation of a residual block which connects the input of the block to its output.

#### Layer
Layers are the basic processing units of any architecture. There is a wide variety of layers ranging from linear layers to activation functions to regularization.

### Dataset
A dataset is a collection of data, typically organized in a structured format, that is used to train the neural networks. Each dataset is composed of individual data points, which may be in the form of rows and columns if represented as a table.

The name argument specifies the name of the dataset file. The path argument defines the directory path where the dataset file is located. The batch size argument sets the number of samples per batch to be loaded during data processing. The random_state argument is used to seed the random number generator.

Then it should be generated specifying the percentages on how to split the dataset into training, validation, and test sets.


```
DatasetGenerator(name="mnist.csv",
                 path="/root/datasets/",
                 batch_size=10,
                 random_state=822).generate(train_proportion=0.7, validation_proportion=0.1, test_proportion=0.2)
```

### Experiment
Experiments are used to define different training runs with varying configurations. Each experiment needs to be set up with an architecture, an optimizer, a loss function and a model saver . This approach allows for the comparison of different setups, helping to identify the optimal configuration for the neural network.

```
Experiment(name="e626",
           architecture=e626,
           optimizer=AdamOptimizer(parameters=e626.parameters(), learning_rate=1.0E-4, betas=(0.9, 0.999), eps=1.0E-8, weight_decay=0.0),
           loss_function=MAELossFunction(),
           saver=ModelSaver("/root/experiment"))
```

### Laboratory

The Laboratory is where the exploration of the optimal neural network occurs. It specifies the parameters and configurations required for the training process, such as the number of epochs, the dataset to use, which experiments to run. The Laboratory also includes strategies depending on the task since it could be a classification or regression model. Additionally, a logger should be defined with a path to write the performance of the experiments, as well as a model loader to get the best model from the explored ones. Lastly, a device could be specified to run the laboratory in a GPU or CPU, being GPU the default if available.

Example:
```
Laboratory(name="MNIST",
           eras=1,
           epochs=10,
           datagen=dataset,
           experiments=experiments,
           strategy=ClassificationStrategy(),
           logger=Logger("C:/Users/Joel/Desktop/test_api/flogo/executions/logger/result.tsv"),
           loader=ModelLoader(),
           device=Device(-1)).explore()
```


## Installation
1. **Clone the Repository in you preferred Python IDE**
2. **Create a Virtual Environment in the Project.**
3. **Install Dependencies**
   - Run the following command in the terminal of the Project ```pip install -r requirements.txt```

With these steps completed, you should have the framework set up and ready to use.

## Examples

### Artificial Neural Network
An artificial neural network (ANN) is a name for a modern feedforward artificial neural network, consisting of fully connected neurons with a nonlinear activation function, organized in at least three layers, notable for being able to distinguish data that is not linearly separable.

Below is an example of an ANN defined using the framework:

```
dataset = DatasetGenerator(name="wine-quality",
                           path="/root/datasets/",
                           batch_size=20,
                           random_state=574).generate(train_proportion=0.6, validation_proportion=0.2, test_proportion=0.2)

architecture = (Architecture("WineQuality")
                    .attach(linearSection([
                                Block([
                                    LinearLayer(in_features=11, out_features=30, dimension=-1, bias=True), 
                                    BatchNormalizationLayer(num_features=30, eps=1.0E-5, momentum=0.3), 
                                    ReLULayer(), 
                                    DropoutLayer(probability=0.5)
                                ]), 
                                Block([
                                    LinearLayer(in_features=30, out_features=10, dimension=-1, bias=True), 
                                    BatchNormalizationLayer(num_features=10, eps=1.0E-5, momentum=0.3), 
                                    ReLULayer(), 
                                    DropoutLayer(probability=0.5)
                                ]), 
                                Block([
                                    LinearLayer(in_features=10, out_features=1, dimension=-1, bias=True), 
                                    ReLULayer()
                                ])
                    ])))

experiments = [Experiment(name="w398",
                          architecture=architecture,
                          optimizer=SGDOptimizer(parameters=architecture.parameters(), learning_rate=1.0E-4, momentum=0.0, dampening=0.0, weight_decay=0.0),
                          loss_function=MSELossFunction(),
                          stopper=EarlyStoper(patience=5, delta=0.001),
                          saver=ModelSaver("/root/experiment"))]

Laboratory(name="WineQuality",
           eras=1,
           epochs=100,
           datagen=dataset,
           experiments=experiments,
           strategy=RegressionStrategy(MSELossFunction()),
           logger=Logger("/executions/logger/result.tsv"),
           loader=ModelLoader(),
           device=Device(0)).explore()
```

### Convolutional Neural Network
A Convolutional Neural Network (CNN) is a type of deep learning model specifically designed for processing structured multidimensional data, such as images. The primary building block of a CNN is the convolutional layer, which applies convolution operations to the input data using a set of filters to extract features. 

Below is an example of a CNN defined using the framework:

```
dataset = DatasetGenerator(name="cats-dogs.csv",
                           path="/root/datasets/",
                           batch_size=10,
                           random_state=463).generate(train_proportion=0.6, validation_proportion=0.2, test_proportion=0.2)

architecture = (Architecture("CatsAndDogs")
                    .attach(convolutionalSection([
                                Block([
                                    ConvolutionalLayer(in_channels=3, out_channels=33, kernel=(3, 3), stride=(2, 2), padding=(0, 0)), 
                                    ReLULayer(), 
                                    MaxPoolLayer(kernel=(5, 5), stride=(4, 4), padding=(0, 0))
                                ]), 
                                Block([
                                    ConvolutionalLayer(in_channels=33, out_channels=16, kernel=(3, 3), stride=(3, 3), padding=(0, 0)), 
                                    ReLULayer(), 
                                    BatchNormalizationLayer(num_features=16, eps=1.0E-5, momentum=0.1), 
                                    MaxPoolLayer(kernel=(3, 3), stride=(2, 2), padding=(1, 1))
                                ])
                    ]))
                    .attach(FlattenLayer(from_dim=3, to_dim=1))
                    .attach(linearSection([
                                Block([
                                    LinearLayer(in_features=64, out_features=2, dimension=-1, bias=True), 
                                    SoftmaxLayer(dimension=-1)
                                ])
                    ])))

experiments = [Experiment(name="i432",
                          architecture=architecture,
                          optimizer=AdamOptimizer(parameters=architecture.parameters(), learning_rate=1.0E-4, betas=(0.9, 0.999), eps=1.0E-8, weight_decay=0.0),
                          loss_function=CrossEntropyLossFunction(),
                          stopper=EarlyStoper(patience=10, delta=0.01),
                          saver=ModelSaver("/root/experiment"))]

Laboratory(name="CatsAndDogs",
           eras=1,
           epochs=50,
           datagen=dataset,
           experiments=experiments,
           strategy=ClassificationStrategy(),
           logger=Logger("/executions/logger/result.tsv"),
           loader=ModelLoader(),
           device=Device(-1)).explore()
```

### Recurrent Neural Network
A Recurrent Neural Network (RNN) is a class of neural networks designed for processing sequential data by maintaining a memory of previous inputs through recurrent connections. This allows RNNs to capture temporal dependencies and patterns in data sequences, making them suitable for tasks like time series prediction, language modeling, and speech recognition.

Below is an example of a RNN defined using the framework:
```
dataset = DatasetGenerator(name="mnist.csv",
                           path="/root/datasets/",
                           batch_size=10,
                           random_state=822).generate(train_proportion=0.7, validation_proportion=0.1, test_proportion=0.2)

architecture = (Architecture("MNIST")
                    .attach(recurrentSection([
                                Block([
                                    LSTMLayer(input_size=28, hidden_size=512, output_type=LSTMLayer.OutputType.EndSequence, num_layer=4, bidirectional=False, dropout=0.0),
                                    FlattenLayer(from_dim=2, to_dim=1), 
                                    ReLULayer()
                                ]), 
                                Block([
                                    LinearLayer(in_features=14336, out_features=300, dimension=-1, bias=True), 
                                    ReLULayer()
                                ]), 
                                Block([
                                    LinearLayer(in_features=300, out_features=75, dimension=-1, bias=True), 
                                    ReLULayer()
                                ]), 
                                Block([
                                    LinearLayer(in_features=75, out_features=10, dimension=-1, bias=True), 
                                    SoftmaxLayer(dimension=-1)
                                ])
])))

experiments = [Experiment(name="e626",
                          architecture=architecture,
                          optimizer=AdamOptimizer(parameters=architecture.parameters(), learning_rate=1.0E-4, betas=(0.9, 0.999), eps=1.0E-8, weight_decay=0.0),
                          loss_function=MAELossFunction(),
                          saver=ModelSaver("/root/experiment"))]

Laboratory(name="MNIST",
           eras=1,
           epochs=10,
           datagen=dataset,
           experiments=experiments,
           strategy=ClassificationStrategy(),
           logger=Logger("/executions/logger/result.tsv"),
           loader=ModelLoader(),
           device=Device(-1)).explore()

```
## License
This project is licensed under the EUPL v1.2 License - see the [LICENSE](LICENSE.txt) file for details.


## Contact

(c) 2024 José Juan Hernández Gálvez 
<br>Github: https://github.com/josejuanhernandezgalvez<br> 
(c) 2024 Juan Carlos Santana Santana 
<br>Github: https://github.com/JuanCarss<br>
(c) 2024 Joel del Rosario Pérez
<br>Github: https://github.com/Joeel71<br>
