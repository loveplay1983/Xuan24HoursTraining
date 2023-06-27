# Xuan's 24 Hours Training!

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
    
    
    