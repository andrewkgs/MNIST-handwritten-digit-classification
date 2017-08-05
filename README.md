MNIST-handwritten-digit-recognition
===
Homework 0 for MLDS course (2017 summer, NTU) <br/>
GOAL: prediction accuracy over 99 %

## MNIST database
<img src="http://knowm.org/wp-content/uploads/Screen-Shot-2015-08-14-at-2.44.57-PM.png" width="40%"> <br/>
Source: http://knowm.org/wp-content/uploads/Screen-Shot-2015-08-14-at-2.44.57-PM.png

## Data source (collected by LeCun et al. and prepared by MLDS2017 TAs)
> Training data (images & labels): <br/>
> http://yann.lecun.com/exdb/mnist/

> Testing data (images): <br/>
> https://mega.nz/#!WlI31LzI!Z6HAOnVIa-AOqQcDMPRKLldZ7Q7nXhsUycT7GY7IND4

## Model description
I use a couple of convolution, activation function(relu), max-pooling technique, and dropout to build this model. <br/>
```
def model(data, train=True):
    conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer1_biases)
    conv = tf.nn.conv2d(hidden, layer2_weights, [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer2_biases)
    pool = tf.nn.max_pool(hidden, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    conv = tf.nn.conv2d(pool, layer3_weights, [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer3_biases)
    conv = tf.nn.conv2d(hidden, layer4_weights, [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer4_biases)
    pool = tf.nn.max_pool(hidden, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    shape = pool.get_shape().as_list()
    reshape = tf.reshape(pool, [shape[0], shape[1] * shape[2] * shape[3]])
    hidden = tf.nn.relu(tf.matmul(reshape, layer5_weights) + layer5_biases)
    if (train):
        hidden = tf.nn.dropout(hidden, 0.75)
    return tf.matmul(hidden, layer6_weights) + layer6_biases
```

## Training parameters and settings
> batch size = 512 <br/>
> training steps = 8000 <br/>
> learning rate = 0.0017(starting rate) with exponential decay after 4000 steps <br/>
> optimizer: AdamOptimizer <br/>

## Results (The score of "B03611026")
* Public score on Kaggle leaderboard (my accuracy = 99.360 %)
<img src="https://github.com/andrewkgs/MNIST-number-recognition/blob/master/public_score.png">

* Private score on Kaggle leaderboard (my accuracy = 99.460 %)
<img src="https://github.com/andrewkgs/MNIST-number-recognition/blob/master/private_score.png">
