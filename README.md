## YOLOv.2 Autonomous Vehicle Image Detection 
### Building an Image Detection Model Using YOLOv.2 Algorithm Using Drive.ai Dataset

### Summary

The goal of this project was to utilize the YOLO image detection algorithm to detect vehicles and provide bounding boxes around the vehicles as described in Redmon et al., 2016 and Redmon and Farhadi, 2016. The dataset was provided by Drive.ai which is a company building software of self-driving vehicles. The detection algorithm consisted of 80 different classes of objects, each with 5 bounding boxes computing probabilities of the object.

<img src= "https://github.com/JeffGoodrich9791/YOLOv2 Autonomous Vehicle Image Detection/blob/master/Bounding_Box_Output.png" />

### Model

Template code is provided in the `YOLO_Autonomous_Driving_Image_Detection.ipynb` notebook file. The layers of the network were constructed using Python 3 and Keras backend in an iPyton Notebook. The identity block was first constructed with 3 components of the main path then adding a shortcut value to pass through a ReLU activation. The first two components consists of Conv2D, BatchNorm, and ReLU activation while the third component has only Conv2D and BatchNorm, with the ReLU activation included in the shortcut path component. 

<img src= "https://github.com/JeffGoodrich9791/ResNet_50_Layer/blob/master/Identity Block.png" />

The first three components of the convolutional block is constructed exactly as the identity block structure. The shortcut component consist of Conv2D as well as BatchNorm, then it is added to the main path and passed through a ReLU activation function. 

<img src= "https://github.com/JeffGoodrich9791/ResNet_50_Layer/blob/master/Convolutional Block.png" />

Once the identity and convolutional blocks are constructed, the ResNet architecture is compiled. The inputs are padded with 3X3 Zero Padding then run through the 50 layer ResNet consisting of 5 stages and a final fully connected (FC) layer. Stage 1 includes a convolution layer, batch normalization, ReLU Activation function, and Max Pooling. Stages 2 through 5 include the previously constructed convolutional and identity block stack. The final layer includes average pooling, flattening, fully connected layer with as softmax function of 6 classes. Details of the entire network describing the architecture, input shape, and parameters can be found in the model summary of the `ResNet_50_Layer.ipynb` notebook file. The following figure describes  the architecture of this neural network. "ID BLOCK" in the diagram stands for "Identity block," and "ID BLOCK x3" means you should stack 3 identity blocks together

<img src= "https://github.com/JeffGoodrich9791/ResNet_50_Layer/blob/master/ResNet Model.png" />

### Run

The model is then run as a model() instance in Keras with AdamOptimizer and categorical crossentropy loss funtion. 

> model = ResNet50(input_shape = (64, 64, 3), classes = 6)

> model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



### Results

The model was trained using a CPU through 2 interations and a batch size of 32. Much better accuracy could have been produced if the session was run using CUDA on a GPU, however; this was not available for training of the model at the time. The results produced with limited number of 2 epochs 
