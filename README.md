HW4 - Understanding CNNs and Generational Adversarial Networks
===================

The assignment consists of training a Generative Adversarial Network on the CIFAR10 dataset as well as a few visualization tasks for better understanding what a CNN is doing. If you missed lecture, you should definitely read https://arxiv.org/pdf/1701.00160.pdf first and check out the lecture slides I posted. If you were at lecture, you should still read it. Everything will make a lot more sense after doing so. The assignment will be split into two main parts. The first is for writing the code defining the full GAN training process. The second is for the visualization tasks I mentioned in the lecture.

The base directory containing the starter files is located under the class directory on BlueWaters and named HW4_GAN. Copy this into your own local directory wherever you want. If you prefer editing with some sort of text editor, you may prefer to copy it to your local computer using the $scp$ command.

Part 1 - GAN Training
-------------
You will be training a generator network and a discriminator network on CIFAR10. The discriminator is simply a classification network similar to what you did in HW2. However, instead of having $10$ outputs for the different classes, it will have $11$. The extra output is for the discriminator to determine if the input is from the generator as opposed to a real image from the actual dataset.

A vector $z$ of dimension $100$ sampled from a random distribution will be input to the generator $G(.)$ and use a series of transposed convolution operators (commonly miscalled deconvolutions) to morph this into a fake image. Both fake images from the generator $G(z)$ and real images $x$ from the dataset will be inputs the discriminator $D(.)$.

The discriminator will try to correctly classify both $D(x)$ and $D(G(z))$. The generator will attempt to fool the discriminator into incorrectly classifying $G(z)$ as one of the $10$ real classes.

The directory structure is already set up for you. There are two python files with some base code and will require a few additions which I'll explain here: $model.py$ and $train.py$.

### $model.py$
This file already contains three functions: $lrelu$, $conv$, and $deconv$. The $lrelu$ is the Leaky ReLU activation function which has been shown to be beneficial for training GANs compared to using ReLU. You will need to define a function for the discriminator and the generator. 

```python
def discriminator(X, keep_prob, is_train=True, reuse=False):
    with tf.variable_scope('discriminator'):
        if reuse:
            tf.get_variable_scope().reuse_variables()


        batch_size = tf.shape(X)[0]
        K = 96
        M = 192
        N = 384

        W1 = tf.get_variable('D_W1', [5, 5, 3, K], 
            initializer=tf.contrib.layers.xavier_initializer())
        B1 = tf.get_variable('D_B1', [K], 
            initializer=tf.constant_initializer())

        W2 = tf.get_variable('D_W2', [5, 5, K, M], 
            initializer=tf.contrib.layers.xavier_initializer())
        B2 = tf.get_variable('D_B2', [M], 
            initializer=tf.constant_initializer())

        W3 = tf.get_variable('D_W3', [5, 5, M, N], 
            initializer=tf.contrib.layers.xavier_initializer())
        B3 = tf.get_variable('D_B3', [N], 
            initializer=tf.constant_initializer())

        W4 = tf.get_variable('D_W4', [3, 3, N, N], 
            initializer=tf.contrib.layers.xavier_initializer())
        B4 = tf.get_variable('D_B4', [N], 
            initializer=tf.constant_initializer())

        W5 = tf.get_variable('D_W5', [4, 4, N, N], 
            initializer=tf.contrib.layers.xavier_initializer())
        B5 = tf.get_variable('D_B5', [N], 
            initializer=tf.constant_initializer())

        W6 = tf.get_variable('D_W6', [N, 10+1], 
            initializer=tf.contrib.layers.xavier_initializer())
        

        conv1 = conv(X, W1, B1, stride=2, name='conv1')
        bn1 = tf.contrib.layers.batch_norm(conv1,
            is_training=is_train,center=True,decay=0.9,updates_collections=None)

        conv2 = conv(tf.nn.dropout(lrelu(bn1), keep_prob),
            W2, B2, stride=2, name='conv2')
        bn2 = tf.contrib.layers.batch_norm(conv2,
            is_training=is_train,center=True,decay=0.9,updates_collections=None)

        conv3 = conv(tf.nn.dropout(lrelu(bn2),keep_prob),
            W3, B3, stride=2, name='conv3')
        bn3 = tf.contrib.layers.batch_norm(conv3,
            is_training=is_train,center=True,decay=0.9,updates_collections=None)

        conv4 = conv(tf.nn.dropout(lrelu(bn3),keep_prob),
            W4, B4, stride=1, name='conv4')
        bn4 = tf.contrib.layers.batch_norm(conv4,
            is_training=is_train,center=True,decay=0.9,updates_collections=None)

        conv5 = conv(lrelu(bn4), W5, B5, stride=1, name='conv5', padding='VALID')
        bn5 = tf.contrib.layers.batch_norm(conv5,
            is_training=is_train,center=True,decay=0.9,updates_collections=None)

        flat = tf.reshape(lrelu(bn5),[batch_size,N])
        output = tf.matmul(flat, W6)

        return tf.nn.softmax(output), output, flat
```

The first $3$ convolutional layers all use $5\times5$ kernels with a stride of $2$ meaning the dimensions transition from $[bs,32,32,3]$&rarr;$[bs,16,16,K]$&rarr;$[bs,8,8,M]$&rarr;$[bs,4,4,N]$.  The output of each is fed into a batch normalization layer followed by a $lrelu$ activation and dropout. By setting updates_collections to $None$, this means any time the network is used for evaluation, it will update the moving averages for the mean and variance if $is\_train$ is set to $True$ and it won't update if $is\_train$ is set to false.

The $conv4$ operation uses only a $3\times3$ kernel ($5\times5$ would be overkill considering the spatial dimensions are only $4\times4$) with a stride of 1. The $conv5$ operation uses $'VALID'$ padding with a $4\times4$ kernel meaning no padding is added resulting in a feature dimension of $[bs,1,1,N]$.  Lastly, this is reshaped to $[bs,N]$ followed by the final fully-connected layer.

Note the usage of the variable $reuse$. This is important. Later on in $train.py$, you will define two separate tensorflow operations for the discriminator, one for real images and one for fake images generated by the generator. Without $reuse$ being $True$ for the second operation, tensorflow would create two separate copies of the discriminator as opposed to creating just one with shared weights between the two.

```python
def generator(Z, keep_prob, is_train=True):
    with tf.variable_scope('generator'):

        batch_size = tf.shape(Z)[0]
        K = 512
        L = 256
        M = 128
        N = 64

        W1 = tf.get_variable('G_W1', [100, 4*4*K],
            initializer=tf.contrib.layers.xavier_initializer())
        B1 = tf.get_variable('G_B1', [4*4*K], initializer=tf.constant_initializer())

        W2 = tf.get_variable('G_W2', [4, 4, L, K], 
            initializer=tf.contrib.layers.xavier_initializer())
        B2 = tf.get_variable('G_B2', [L], 
            initializer=tf.constant_initializer())

        W3 = tf.get_variable('G_W3', [6, 6, M, L], 
            initializer=tf.contrib.layers.xavier_initializer())
        B3 = tf.get_variable('G_B3', [M], 
            initializer=tf.constant_initializer())

        W4 = tf.get_variable('G_W4', [6, 6, N, M], 
            initializer=tf.contrib.layers.xavier_initializer())
        B4 = tf.get_variable('G_B4', [N], 
            initializer=tf.constant_initializer())

        W5 = tf.get_variable('G_W5', [3, 3, N, 3], 
            initializer=tf.contrib.layers.xavier_initializer())
        B5 = tf.get_variable('G_B5', [3], 
            initializer=tf.constant_initializer())

        Z = lrelu(tf.matmul(Z, W1) + B1)
        Z = tf.reshape(Z, [batch_size, 4, 4, K])
        Z = tf.contrib.layers.batch_norm(Z,
            is_training=is_train,center=True,decay=0.9,updates_collections=None)

        deconv1 = deconv(Z, 
            W2, B2, shape=[batch_size, 8, 8, L], stride=2, name='deconv1')
        bn1 = tf.contrib.layers.batch_norm(deconv1,
            is_training=is_train,center=True,decay=0.9,updates_collections=None)

        deconv2 = deconv(lrelu(bn1), 
            W3, B3, shape=[batch_size, 16, 16, M], stride=2, name='deconv2')
        bn2 = tf.contrib.layers.batch_norm(deconv2,
            is_training=is_train,center=True,decay=0.9,updates_collections=None)

        deconv3 = deconv(lrelu(bn2), 
            W4, B4, shape=[batch_size, 32, 32, N], stride=2, name='deconv3')
        bn3 = tf.contrib.layers.batch_norm(deconv3,
            is_training=is_train,center=True,decay=0.9,updates_collections=None)

        conv4 = conv(lrelu(bn3), W5, B5, stride=1, name='conv4')
        output = tf.nn.tanh(conv4)

        return output
```
The generator starts with a fully connected layer reshaping the $[bs,100]$ random input $z$ to $[bs,4,4,512]$. This is followed by $3$ transposed convolutions resulting in a $[bs,32,32,N]$ tensor. The final layer is a regular convolution simply reducing the last dimension to $3$ so it is in the shape of an image. Note the $tanh$ activation function on the output. This is to bound the generated image between $-1$ and $1$ such that it remains in the same range as the true images $x$ (which were rescaled to be between $-1$ and $1$).

Note the chosen kernel sizes are even for the transposed convolution layers with a stride of $2$ as opposed to usually choosing odd sized kernels typically seen with convolutional layers. This is to prevent certain artifacts from appearing in the generated images. The kernel size for transposed convolutions should always be divisible by the stride. See https://distill.pub/2016/deconv-checkerboard/ for a better visualization of what I mean by artificats.

### $train.py$
The first $279$ lines of $train.py$ can go untouched. This is nearly identical to the data augmentation I provided previously for HW2 for the which the explanation can be found on Piazza. We can now start defining the necessary tensorflow operations for training our model.

```python
with tf.variable_scope('placeholder'):
    X = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 3])
    z = tf.placeholder(tf.float32, [None, 100])  # noise
    y = tf.placeholder(name='label',dtype=tf.float32,shape=[batch_size,10])
    keep_prob = tf.placeholder(tf.float32 ,shape=())
    is_train = tf.placeholder(tf.bool ,shape=())
```
The only thing new here is the placeholder for $z$, the random input to the generator.

```python
with tf.variable_scope('GAN'):
    G = generator(z, keep_prob=keep_prob, is_train=is_train)

    D_real, D_real_logits, flat_features = discriminator(X,
        keep_prob=keep_prob, is_train=is_train, reuse=False)
    D_fake, D_fake_logits, flat_features = discriminator(G,
        keep_prob=keep_prob, is_train=is_train, reuse=True)
```

It is important to define two separate operations for the discriminator with the second one having a $reuse$ value to be $True$. As mentioned previously, this prevents tensorflow from creating two separate networks and instead creates one discriminator with shared weights. 

```python
with tf.variable_scope('D_loss'):
    real_label = tf.concat([0.89*y,tf.zeros([batch_size,1])],axis=1)
    real_label += 0.01*tf.ones([batch_size,11])

    fake_label = tf.concat([tf.zeros([batch_size,10]),
        tf.ones([batch_size,1])],axis=1)

    d_loss_real = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
        logits=D_real_logits,labels=real_label))
    
    d_loss_fake = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
        logits=D_fake_logits,labels=fake_label))

    d_loss = d_loss_real + d_loss_fake
```
The placeholder $y$ is only for the basic $10$ classes and a necessary $0$ is appended to the end for the fake class.  As mentioned in the NIPS tutorial by Ian Goodfellow, it is beneficial to apply label smoothing to prevent the discriminator from becoming overconfident. The true label is given a value of $0.89$ with all incorrect labels being assigned $0.01$. The total loss for $D$ is now a combination of two losses. $d\_loss\_real$ refers to the same loss used in the last assignment. This is the network trying to learn how to classify images. $d\_loss\_fake$ is the new loss based on whether it correctly identifies the generated images $G(z)$ as fake. Every time we perform an update on the trainable variables of $D$, we will be feeding it both a batch of real images and a batch of generated images.

```python
with tf.variable_scope('G_loss'):
    g_loss = tf.reduce_mean(tf.log(D_fake[:,-1]))
```
The loss for the generator is based on the predicted confidence from the discriminator for all of the fake images. Normally we would like to minimize the negative log likelihood as a typical loss functions. However, the generator actually wants to maximize the negative log likelihood (it wants the discriminator to incorrectly identify the images). Tensorflow is set up to perform minimization already which explains why the negative was removed. Minimizing the log likelihood is the same as maximizing the negative log likelihood.

```python
with tf.variable_scope('accuracy'):
    correct_prediction = tf.equal(
        tf.argmax(D_real[:,:-1],1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(
        tf.cast(correct_prediction,tf.float32))

tvar = tf.trainable_variables()
dvar = [var for var in tvar if 'discriminator' in var.name]
gvar = [var for var in tvar if 'generator' in var.name]
```

These are quite basic. Notice the last output (dimension $11$) is ignored for calculating the correct prediction. In our test scenario for accuracy, we care only about the accuracy of the $10$ real classes. 

```python
with tf.name_scope('train'):
    d_train_step = tf.train.AdamOptimizer(
        learning_rate=0.5*(1e-4), beta1=0.5).minimize(d_loss, var_list=dvar)
    g_train_step = tf.train.AdamOptimizer(
        learning_rate=1e-4, beta1=0.5).minimize(g_loss, var_list=gvar)
```

Lastly we define the two training steps. Each train step has its own corresponding loss as well as its own variables to update.

```python
same_input = np.random.uniform(-1., 1., [64, 100])
same_input = same_input/np.sqrt(np.sum(same_input**2,axis=1))[:,None]
```

Although we will sample a new $z$ every iteration, it is helpful to save one particular sample. This can be used to periodically save the fake images generated by G as to monitor progress. After training, it is interesting to see how these specific samples evolve over time. The second line is simply rescaling the vector to have magnitude one. Instead of sampling from a $100$ dimensional unit hypercube, the rescaling essentially makes it a sample from a $100$ dimensional sphere.

```python

sess = tf.InteractiveSession()

init = tf.global_variables_initializer()
sess.run(init)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)
tf.train.start_queue_runners()
```

This is common stuff. The queue runners are for syncing up the cpu threads used to load the images in.

```python
num_img = 0
d_loss_real_p = 0
d_loss_fake_p = 0
g_loss_p = 0
t = time.time()
for i in range(0,250000):
    # update D
    X_batch, labels_batch = sess.run([images, labels])
    z_batch = np.random.uniform(-1., 1., [batch_size, 100])
    z_batch = z_batch/np.sqrt(np.sum(z_batch**2,axis=1))[:,None]
    y_batch = np.zeros((batch_size,10))
    y_batch[np.arange(batch_size),labels_batch] = 1

    _, d_loss_real_p, d_loss_fake_p, accuracy_p = sess.run(
        [d_train_step, d_loss_real, d_loss_fake, accuracy], 
        feed_dict={X: X_batch, z: z_batch, 
        y: y_batch, keep_prob:0.5,  is_train:True})

    # update G
    z_batch = np.random.uniform(-1., 1., [batch_size, 100])
    z_batch = z_batch/np.sqrt(np.sum(z_batch**2,axis=1))[:,None]
    _, g_loss_p = sess.run([g_train_step, g_loss], 
        feed_dict={X: X_batch, z: z_batch, keep_prob:0.5, is_train:True})

    # monitor progress
    if i % 20 == 0:
        print('time: %f epoch:%d g_loss:%f d_loss_real:%f d_loss_fake:%f accuracy:%f'
            % (float(time.time()-t), i, g_loss_p, 
            d_loss_real_p, d_loss_fake_p, accuracy_p))
        t = time.time()

    # every 500 batches, save the generator output
    if i % 500 == 0:
        samples = sess.run(G, 
            feed_dict={z: same_input, keep_prob:1.0, is_train:False})
        samples += 1.0
        samples /= 2.0
        fig = plot(samples)
        plt.savefig('output/%s.png' % str(num_img).zfill(3), bbox_inches='tight')
        num_img += 1
        plt.close(fig)

    # every 500 batches, try the test dataset
    if i % 500 == 0:
        test_accuracy = 0.0
        accuracy_count = 0
        for j in xrange(50):
            X_batch, labels_batch = sess.run([images_test,labels_test])
            y_batch = np.zeros((batch_size,10))
            y_batch[np.arange(batch_size),labels_batch] = 1

            accuracy_p = sess.run([accuracy], 
                feed_dict={X: X_batch, y: y_batch, keep_prob:1.0, is_train:False})

            test_accuracy += accuracy_p[0]
            accuracy_count += 1
        test_accuracy = test_accuracy/accuracy_count
        print('TEST:%f' % test_accuracy)
```

Here is the main training loop. I recommend training for $250,000$ iterations. For each iteration, there is one update for the discriminator and one update for the generator. Every $20$ iterations, the $3$ loss functions and the training accuracy are printed. During training, the generator loss and the discriminator fake loss should be hovering around $0.70$ which corresponds with discriminator outputting a confidence of around $50\%$. If the discriminator becomes highly confident of classifying the fake images, this will cause the generator loss to approach $0$ and stop learning. The discriminator real loss will show the same typical pattern as training a regular NN on a classification task. It will start high and gradually decrease.

Every $500$ iterations, the generator output is saved and the discriminator is evaluated on the test set. 

```python
all_vars = tf.global_variables()
dvars = [var for var in all_vars if 'discriminator' in var.name]
dvars = [var for var in dvars if 'Adam' not in var.name]
saver = tf.train.Saver(dvars)
saver.save(sess, 'GAN/discriminator/model')

gvars = [var for var in all_vars if 'generator' in var.name]
gvars = [var for var in gvars if 'Adam' not in var.name]
saver = tf.train.Saver(gvars)
saver.save(sess, 'GAN/generator/model')
```

These are saving not only the trainable variables, but all of the variables associated with both networks. The batch normalization layers have $4$ variables each ($gamma$, $beta$, $moving\_mean$, and $moving\_variance$). $Beta$ and $gamma$ are trained via back propagation and, therefore, are contained in $tf.trainable\_vars()$. The $moving\_mean$ and $moving\_variance$ are updated separate of back propagation and therefore they are not considered trainable variables. 

This should be all that is necessary for training the GAN! I've selected (what I hope) to be safe hyperparameters meaning the model will not experience mode collapse and will successfully train. However, there is a chance these networks can get stuck and fail. You may have to run it again if that's the case. Let me know if this happens as I would be curious to see.

Part 2 - Visualizations
-------------
This section consists of four different tasks which will generate plots to turn in. The first $279$ lines of $visualization.py$ is the same as $train.py$. Technically, you could do this part at the bottom of $train.py$ or split it up into separate files or anything you want. I'll give you the code and techniques, but let you worry about how you go about executing the code. Some of it will execute very fast so it may be easier to do this part from an interactive session.

The following code will be used to slightly perturb real images to produce highly confident incorrect outputs, slightly perturb random images to produce highly confident incorrect outputs, create images attaining very high activations in the output layer, and create images attaining very high activations in the layer before the output layer.

```python
with tf.variable_scope('placeholder'):
    X = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 3])
    y = tf.placeholder(name='label',dtype=tf.float32,shape=[None,10])
    keep_prob = tf.placeholder(tf.float32 ,shape=())
    is_train = tf.placeholder(tf.bool ,shape=())
    
with tf.variable_scope('GAN'):
    D, D_logits, flat_features = discriminator(X, 
        keep_prob=keep_prob, is_train=is_train, reuse=False)

with tf.variable_scope('D_loss'):
    label = tf.concat([y,tf.zeros([batch_size,1])],axis=1)
    d_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
        logits=D_logits,labels=label))

with tf.variable_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(D[:,:-1],1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
```

There is nothing particularly unusual here. It is a simplified version of $train.py$. We only need the discriminator for this part and not the generator.

```python

with tf.name_scope('gradients'):
    grad_loss_over_X = tf.gradients(d_loss, X)[0]

    grad_features_over_X = tf.gradients(
        tf.reduce_mean(tf.diag_part(flat_features[0:64,0:64])),X)[0]
    grad_logit_over_X = tf.gradients(
        tf.reduce_mean(tf.diag_part(D_logits[0:10,0:10])),X)[0]
```
Instead of directly using an optimizer, we will use tensorflow's operation for computing gradients. The first gradient is the same as what would be used for training the network except it's propagated all the way back to the input. The second one is a bit more tricky. The tensor $flat\_features$ is a tensor of dimension $[bs,384]$. This is right before the output layer. In this case, we are going to input a batch size of $64$ and extract the square matrix from tensor. The operation $tf.diag\_part$ extracts the diagonal elements of the matrix. This is essentially grabbing one feature for each image in the batch (the $i^{th}$ image in the batch will correspond with the $i^{th}$ feature). This operation will return the gradient of the feature with respect to the input which can be used to alter the image and create a high feature output. The last gradient is nearly the same except it uses the logits instead. In this case, we will input a batch size of $10$.

```python
dvar = tf.global_variables()
saver = tf.train.Saver(dvar)

sess = tf.InteractiveSession()

init = tf.global_variables_initializer()
sess.run(init)

#saver.restore(sess,tf.train.latest_checkpoint('GAN/discriminator/'))
saver.restore(sess,tf.train.latest_checkpoint('discriminator/'))

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)
tf.train.start_queue_runners()
```

This will load the model and start loading images. I will be giving you a pretrained model (trained without a generative network) and you will be training your own model with the GAN. This code should work for both models.

### Perturb Real Images

```python
X_batch, labels_batch = sess.run([images_test, labels_test])

real_labels = labels_batch
alternate_labels = labels_batch + 1
alternate_labels[alternate_labels>=10]=0

y_batch_real = np.zeros((batch_size,10))
y_batch_real[np.arange(batch_size),real_labels] = 1

y_batch_alternate = np.zeros((batch_size,10))
y_batch_alternate[np.arange(batch_size),alternate_labels] = 1
```

A batch of images from the test set is used along with it's real label. Additionally, an alternate label is made by adding one to the real label (loops back to $0$ if $\geq10$).

```python
gradient, pred, logit, correct = sess.run(
    [grad_loss_over_X,D,D_logits,correct_prediction], 
    feed_dict={X: X_batch, y: y_batch_real, keep_prob:1.0, is_train:False})
```

This is simply to store the variables $pred$, $logit$, and $correct$, to see how they are altered by the perturbation.

```python
gradient = sess.run(grad_loss_over_X, 
    feed_dict={X:X_batch, y: y_batch_alternate, keep_prob:1.0, is_train:False})

gradient_image = (gradient - np.min(gradient))/(np.max(gradient)-np.min(gradient))
fig = plot(gradient_image)
plt.savefig('gradient.png', bbox_inches='tight')
plt.close(fig)
```

Here is where we use the gradient based on the alternative label. The gradient is scaled from $-1$ to $1$ and plotted as an image. 

```python
gradient[gradient>0.0] = 1.0
gradient[gradient<0.0] = -1.0

X_batch_modified = X_batch - 10.0*0.007843137*gradient
X_batch_modified[X_batch_modified>1.0] = 1.0
X_batch_modified[X_batch_modified<-1.0] = -1.0

pred_alternate, logit_alternate, correct_alternate = sess.run(
    [D,D_logits,correct_prediction],
    feed_dict={X:X_batch_modified, y: y_batch_real, keep_prob:1.0, is_train:False})
  
X_batch += 1.0
X_batch /= 2.0
X_batch[X_batch>1.0] = 1.0
X_batch[X_batch<0.0] = 0.0
fig = plot(X_batch)
plt.savefig('X.png', bbox_inches='tight')
plt.close(fig)

X_batch_modified += 1.0
X_batch_modified /= 2.0
X_batch_modified[X_batch_modified>1.0] = 1.0
X_batch_modified[X_batch_modified<0.0] = 0.0
fig = plot(X_batch_modified)
plt.savefig('X_alternate.png', bbox_inches='tight')
plt.close(fig)
```

The original image is scaled from $0$ to $255$ to $-1$ to $1$ meaning we have a resolution of $0.0078$. We modify the image by $10$ in the opposite direction of the gradient for each pixel (trying to minimize the loss). We now get the variables $pred\_alternate$, $logit\_alternate$ and $correct\_alternate$ which can be used to compare the output before and after the modification. The images before and after the modification are saved as well. Use the variables to demonstrate how the prediction changed even though the image looks nearly the same from our perspective.

### Perturb Image of Noise

```python
X_batch = np.random.normal(0.0,1.0, [64, 32, 32, 3])
X_batch = ((X_batch - np.min(X_batch))/(np.max(X_batch)-np.min(X_batch))*2.0) - 1.0
```

Use this batch of noise to get a similar output to the one above. Show how the output changed after the modification.

### High Activation Images for Output Layer

```python
X_batch,_ = sess.run([images_test, labels_test])
X_batch = np.mean(X_batch,axis=0)[np.newaxis,:,:,:]
X_batch = np.repeat(X_batch,10,axis=0)

X_batch_modified = 1.0*X_batch

t1 = time.time()
for i in xrange(500):
    gradient, logits = sess.run([grad_logit_over_X, D_logits],
        feed_dict={X:X_batch_modified, keep_prob:1.0, is_train:False})

    X_batch_modified = X_batch_modified + 0.3*gradient - 0.003*X_batch_modified

    X_batch_modified[X_batch_modified>1.0] = 1.0
    X_batch_modified[X_batch_modified<-1.0] = -1.0
    if i % 50 == 0:
        print(i, logits[np.arange(10),np.arange(10)])

X_batch_save = X_batch_modified*1.0
X_batch_save += 1.0
X_batch_save /= 2.0
X_batch_save[X_batch_save>1.0] = 1.0
X_batch_save[X_batch_save<0.0] = 0.0
fig = plot_classes(X_batch_save)
plt.savefig('classes.png', bbox_inches='tight')
plt.close(fig)
```

This is for calculating a fake image with a high output for each class. Run this code and explain how it works. Try training for longer than $500$ iterations and see the output. Try modifying the learning rate for the gradient and the weight decay.

### High Activation Image for Intermediate Features

```python
X_batch,_ = sess.run([images_test, labels_test])
X_batch = np.mean(X_batch,axis=0)[np.newaxis,:,:,:]
X_batch = np.repeat(X_batch,64,axis=0)

X_batch_modified = 1.0*X_batch

t1 = time.time()
for i in xrange(500):
    gradient, features = sess.run([grad_features_over_X, flat_features],
        feed_dict={X:X_batch_modified, keep_prob:1.0, is_train:False})

    X_batch_modified = X_batch_modified + 1.0*gradient - 0.001*X_batch_modified

    X_batch_modified[X_batch_modified>1.0] = 1.0
    X_batch_modified[X_batch_modified<-1.0] = -1.0
    if i % 50 == 0:
        print(i)

X_batch_save = X_batch_modified*1.0
X_batch_save += 1.0
X_batch_save /= 2.0
X_batch_save[X_batch_save>1.0] = 1.0
X_batch_save[X_batch_save<0.0] = 0.0
fig = plot(X_batch_save)
plt.savefig('features.png', bbox_inches='tight')
plt.close(fig)
```

This is for calculating a fake image with a high output for the first $64$ features of the layer before the output. Run this code and explain how it works. Try training for longer than $500$ iterations and see the output. Try modifying the learning rate for the gradient and the weight decay.


What to turn in
-------------
Write a short report with pictures included for the four different visualization parts. Make sure to include all the saved images with a short description of what you see. Repeat the visualization part for both the network I provided as well as the network you trained and compare the two. Include a brief description of how the code works for calculating the high activation images.

Create an animated gif or video of the saved images from the generator to show how it evolved over time.

Place the above items as well as your completed code into a directory, compress it, and submit.

Please let me know if there are any questions or confusion. The extra credit mentioned in the lecture can be included here as well if you choose to do it.
