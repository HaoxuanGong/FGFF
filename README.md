# FGFF
The implementation for the Fine-Grained Forward Forward Algorithm


## How to Run

1. Install all the necessary dependencies
2. Run the sup_self.py file
```python
python sup_self.py
```

## How to test the model on different network architectures
- To adjust the network's architecture, you need to modify codes in both `sup_self.py` and `layer_self.py`
- In `sup_self.py`, modify this line of code:
```python
network = Network([784, 500, 500]).cuda()
```
- In `layer_self.py`, modify this line of code:
```python
self.layer_weights = nn.Parameter(torch.ones(2, 500)) # the significance matrix
```
- Make sure the dimension of the significance matrix is equivalent to the **number of hidden layers** * the **number of neurons in each hidden layer**
- Test different learning rates for the significance matrix in this line of code in `layer_self.py`:
```python
self.weight_optimizer = Adam([self.layer_weights], lr=0.15)
```
- To test the performance of the Forward Forward algorithm, adjust the learning rate for the significance matrix optimizer to 0
- Test different learning rates for the network weightings in this line of code in `layer_self.py`:
```python
self.learning_rate = 0.06
```

## How to test the model on different datasets
- Modify this line of code in `sup_self.py`:
```python
training_data_loader, testing_data_loader = load_MNIST_data()
```


## Results

The performance of the Fine-Grained Forward Forward (FGFF) algorithm was compared with the Forward Forward (FF) algorithm on the MNIST and Fashion-MNIST datasets. The test accuracies are summarized below.

### MNIST Dataset

| Model         | FF       | FF (lr tuned) | FGFF     |
|---------------|----------|---------------|----------|
| 2x50 Relu     | 63.85%   | 74.71%        | 85.73%   |
| 2x500 Relu    | 66.97%   | 85.82%        | 90.83%   |
| 2x1000 Relu   | 68.41%   | 85.83%        | 89.58%   |

**Table I:** Test accuracies (%) on MNIST dataset for forward-forward (FF) and fine-grained forward-forward (FGFF) algorithms.

The results on MNIST are summarized in Table I. All models were trained for 200 epochs. FF and FGFF used a learning rate of 0.06. FGFF consistently outperformed FF across all models.

### Fashion-MNIST Dataset

| Model         | FF       | FF (lr tuned) | FGFF     |
|---------------|----------|---------------|----------|
| 2x50 Relu     | 53.16%   | 60.76%        | 73.28%   |
| 2x500 Relu    | 50.71%   | 52.24%        | 75.63%   |
| 2x1000 Relu   | 49.42%   | 51.64%        | 77.18%   |

**Table II:** Test accuracies (%) on Fashion-MNIST dataset for forward-forward (FF), fine-grained forward-forward (FGFF), and tuned FF models with various architectures.

The results in Table II demonstrate that increasing the number of neurons in each layer tends to improve model performance on the Fashion-MNIST dataset, similar to the observations in Table I for MNIST.

### CIFAR-10 Dataset

| Algorithm | Train Acc (%) | Test Acc (%) |
|-----------|---------------|--------------|
| FF        | 38.33%        | 37.67%       |
| FGFF      | 50.73%        | 42.67%       |

**Table III:** Test accuracies (%) on CIFAR-10 dataset for forward-forward (FF) and fine-grained forward-forward (FGFF) algorithms.

## Contributors
- [Bruce Lee](https://github.com/tli389)
- [James Gong](https://github.com/HaoxuanGong)
