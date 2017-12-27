import tensorflow as tf


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data", one_hot=True, reshape=True, validation_size=0)


learning_rate = 0.004
training_epochs = 30
batch_size = 100
display_step = 1


n_hidden_1 = 256 # 1st layer 
n_hidden_2 = 256 
n_hidden_3 = 256 
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


def multilayer_perceptron(x, weights, biases):
     
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1']) #matmul instead of * 
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
       
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3']) 
    layer_3 = tf.nn.relu(layer_3)
    
    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']    

    return out_layer

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),    #784x256
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])), #256x256
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])), #256x256
    'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes]))  #256x10
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),             #256x1
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),             #256x1
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),             #256x1
    'out': tf.Variable(tf.random_normal([n_classes]))              #10x1
}


pred = multilayer_perceptron(x, weights, biases)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)
cross_entropy = tf.reduce_mean(cross_entropy)*100


optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)


init = tf.initialize_all_variables()


with tf.Session() as sess:
    sess.run(init)


    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)

        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)

            _, c = sess.run([optimizer, cross_entropy], feed_dict={x: batch_x, y: batch_y})
            print(_, c)

            avg_cost += c / total_batch

        if epoch % display_step == 0:
            print ("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
    print("Learning Finished!")


    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print ("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


