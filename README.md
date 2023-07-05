# Xuan's 24 Hours Training!

### PyTorch resource
https://github.com/pytorch/pytorch


### PyTorch libraries, and others

- torch.nn
    - torch.nn.BCELossWithLogits (Binary classification) 
    - torch.nn.BCELoss           (Binary classification)
    - torch.nn.CrossEntropyLoss  (Mutli-class classification)
    - torch.nn.L1Loss            (Regression)
    - torch.nn.MSELoss           (Regression)

Contains all of the building blocks for computational graphs (essentially a series of computations executed in a particular way).

- torch.nn.Parameter

Stores tensors that can be used with nn.Module. If requires_grad=True gradients (used for updating model parameters via gradient descent) are calculated automatically, this is often referred to as "autograd".

- torch.nn.Module

The base class for all neural network modules, all the building blocks for neural networks are subclasses. If you're building a neural network in PyTorch, your models should subclass nn.Module. Requires a forward() method be implemented.

- torch.optim

Contains various optimization algorithms (these tell the model parameters stored in nn.Parameter how to best change to improve gradient descent and in turn reduce the loss).
    - torch.optim.SGD()     (Classification, regression, many others)
    - torch.optim.Adam()    (Classification, regression, many others)



- def forward()

All nn.Module subclasses require a forward() method, this defines the computation that will take place on the data passed to the particular nn.Module (e.g. the linear regression formula above).


- Metrics
    - torchmetrics.Accuracy()
    - sklearn.metrics.accuracy_score()

    - torchmetrics.Precsion
    - sklearn.metrics.precision_score()
    
    - torchmetrics.Recall()
    - sklearn.metrics.recall_score()
    
    - torchmetrics.F1Score()
    - sklearn.metrics.f1_score()
    
    - torchmetrics.ConfusionMatrix()
    - sklearn.metrics.plot_confusion_matrix()
    
    - sklearn.metrics.classification_report()
    
    - torch.distributioned.elastic.metrics 
    
    
- torchvision
    - torchvision.datasets
    - torchvision.models
    - torchvision.transforms
    - torch.utils.data.Dataset
    - torch.utils.data.DataLoader()
    
    
### Megic functions in Jupyter-notebook
https://towardsdatascience.com/9-magic-command-to-enhance-your-jupyter-notebook-experience-101fb5f3a84#:~:text=True%20to%20its%20name%2C%20Magic,Magic%20Command%20in%20this%20article.   
https://www.dataquest.io/blog/jupyter-notebook-tips-tricks-shortcuts/


`% vs %%`
- %time measures execution time of the next line.
- %%time measures execution time of the whole cell.


- %who   
> This command is a Magic command that would show all the available variables you had in your Jupyter Notebook Environment
```python
import seaborn as sns
df = sns.load_dataset('mpg')
a = 'simple'
b = 2

%who str
```

- %timeit
> This magic command was used to evaluate the code execution speed by running it multiple times and produce the average + standard deviation of the execution time
```python
import numpy as np
%timeit np.random.normal(size=1000)
```


- %store
> if you work on a project in one notebook and want to pass around the variables you had into another notebook. You do not need to pickle it or save it in some object file. What you need is to use the %store magic command.
```python
%store df
%store -r df
```

- %prun
> %prun is a specific magic command to evaluate how much time your function or program to execute each function.
```python
%prun sns.load_dataset('mpg')
```

- %history or %hist
> %history magic command to see the log of your activity and trace back what you already did.
```python
%history
```

- %pinfo
> working with a new object or packages, you want to get all the detailed information
```python
%pinfo df
```

- %%writefile
```python
%%writefile test.py
def number_awesome(x):
    return 9
```

- %pycat
>  reading the Python file into your Jupyter Notebook
```python
%pycat test.py
```

- %quickref
>  explains all the magic command that exists in the Jupyter Notebook with detail.


### Training tricks
- multi-gpu
https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html  
https://pytorch.org/tutorials/beginner/ddp_series_multigpu.html 
https://www.run.ai/guides/multi-gpu/pytorch-multi-gpu-4-techniques-explained
https://www.aime.info/blog/en/multi-gpu-pytorch-training/

### Other ML study resource
- d2l
https://d2l.ai/
