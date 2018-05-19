# Homework 2 for Deep Learning

## Environment

I have test it on different machines, so it's a little complicated.
* ubuntu 16.04 LTS
* python3.5.2 or python3.6
* tensorflow 1.4.1 or 1.7.0
* GPU: Tesla P100

### CNN for Image Recognition

In this task, I construct a 4-layer CNN to predict the label of [Food-11 Dataset](https://mmspg.epfl.ch/food-image-datasets).
I used tensorflow high level api Dataset to read the image and do some preprocessing.
The detailed implementation can be seen in [cnn_train_fast.py](cnn_train_fast.py).
By specified the directory where you put the images, you can start to train the model.
After training it will generate training curve and training accuracy, and some model meta data.

![](images/model2_curve.png)
![](images/model2_acc.png)

Use [plot_model.py](plot_model.py) to visualize the distribution of model weights and bias.
Be sure to specified the meta files directory.

![](images/conv_weights.png)
![](images/conv_bias.png)

[feature_map.py](feature_map.py) will compute the model confusion matrix evaluated on evaluation data, 
display some recognition results and visualize hidden features of each layer.

![](images/confusion%20matrix.png)
![](images/correct1.png)
![](images/correct1_conv1.png)
![](images/false1.png)
![](images/false1_conv1.png)

### RNN for Language Model

In this task I construct a 2-layer RNN as character-level language model, 
and trained it on [The Complete Works of William Shakespeare](http://shakespeare.mit.edu/works.html).
After training, it will generate training curve and training error as usual, and some random generated words with starting text _Asuka_.

![](images/lstm_Training%20curve.png)
![](images/lstm_Training%20error.png)

```
Start training ...
Epochs: 1, loss: 2.4919, acc: 0.3128, val_acc: 0.4112
Epochs: 2, loss: 1.9383, acc: 0.4446, val_acc: 0.4599
Epochs: 3, loss: 1.7829, acc: 0.4877, val_acc: 0.4841
Epochs: 4, loss: 1.6837, acc: 0.5163, val_acc: 0.5016
Epochs: 5, loss: 1.6151, acc: 0.5355, val_acc: 0.5148
Epochs: 6, loss: 1.5657, acc: 0.5485, val_acc: 0.5236
Epochs: 7, loss: 1.5281, acc: 0.5584, val_acc: 0.5303
Epochs: 8, loss: 1.4986, acc: 0.5660, val_acc: 0.5369
Epochs: 9, loss: 1.4746, acc: 0.5721, val_acc: 0.5425
Epochs: 10, loss: 1.4548, acc: 0.5772, val_acc: 0.5470
Epochs: 11, loss: 1.4379, acc: 0.5814, val_acc: 0.5506
Epochs: 12, loss: 1.4235, acc: 0.5850, val_acc: 0.5531
Epochs: 13, loss: 1.4109, acc: 0.5882, val_acc: 0.5550
Epochs: 14, loss: 1.3996, acc: 0.5910, val_acc: 0.5564
Epochs: 15, loss: 1.3897, acc: 0.5935, val_acc: 0.5581
Epochs: 16, loss: 1.3816, acc: 0.5953, val_acc: 0.5593
Epochs: 17, loss: 1.3728, acc: 0.5977, val_acc: 0.5606
Epochs: 18, loss: 1.3654, acc: 0.5995, val_acc: 0.5619
Epochs: 19, loss: 1.3586, acc: 0.6012, val_acc: 0.5631
Epochs: 20, loss: 1.3523, acc: 0.6027, val_acc: 0.5640
Epochs: 21, loss: 1.3466, acc: 0.6042, val_acc: 0.5650
Epochs: 22, loss: 1.3408, acc: 0.6056, val_acc: 0.5653
Epochs: 23, loss: 1.3357, acc: 0.6069, val_acc: 0.5657
Epochs: 24, loss: 1.3308, acc: 0.6082, val_acc: 0.5655
Epochs: 25, loss: 1.3264, acc: 0.6094, val_acc: 0.5664
Epochs: 26, loss: 1.3222, acc: 0.6104, val_acc: 0.5669
Epochs: 27, loss: 1.3182, acc: 0.6114, val_acc: 0.5675
Epochs: 28, loss: 1.3144, acc: 0.6123, val_acc: 0.5678
Epochs: 29, loss: 1.3109, acc: 0.6133, val_acc: 0.5680
Epochs: 30, loss: 1.3075, acc: 0.6141, val_acc: 0.5689
Asukal their sons
and to the senseless off and things that seems
Which they are so as well as well as well.
Therefore, to see the street of the strength of her face.

PETRUCHIO:
What say you?

PETRUCHIO:
What is the news of this? What shall we do?

LUCIANA:
And that we shall be satisfied to his daughter,
To see the sea of true consent than thine,
And then, and that the sea was stay'd to do
Assemb a secret that we here, a seal
That they are struckingly to the duke of the
present constable and a most p
Time cost: 2:41:15
```
