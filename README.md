# AI-OCR-96
Persian Character based OCR based on Deep Learning. Implemented using [Tensorflow](https://github.com/tensorflow/tensorflow/).

----

## Version 1: Deep Perceptron Neywork 

## Execute with English MNIST

  - The English verion of the code can be easily executed. The Tensorflow library will automatically download the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.
  - The entire code is self-contained and is only dependant on a single library, namely Tensorflow. 
  - The following parameters can be configured inside the code: 
  
```
learning_rate = 0.004
training_epochs = 30
batch_size = 100
```

The previous values are the ones proven to be the best and the most efficient based on our experiments. 


-----

## Version 2: Deep Convolutional Neural Network 

- Use [This line]() to choose mnist or hoda dataset. 
- In order to use Hoda, use [this link](https://mega.nz/#!qU5lBI7R!ne9bWWiPPGcPL-b4G1-s4i-1ca2nh4lLceFuafJHY8E) to download them. Extract the zip archive into the root directory of this project. 4 Files with name `test_dump_img_40.npy` should be extracted. 
- A utility file, `parser.cdb.py` is provided. You can use it to extract your own cdb files and dump them to `.npy`. 