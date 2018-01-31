## YOLOv.2 Autonomous Vehicle Image Detection 
### Building an Image Detection Model Using YOLOv.2 Algorithm Using Drive.ai Dataset

### Summary

The goal of this project was to utilize the YOLO image detection algorithm to detect vehicles and provide bounding boxes around the vehicles as described in Redmon et al., 2016 and Redmon and Farhadi, 2016. The dataset was provided by Drive.ai which is a company building software of self-driving vehicles. The detection algorithm consists of 80 different classes of objects, each with 5 bounding boxes computing probabilities of the object. For efficiency we used a model with pre-trained weights which come from the official YOLO website, and were converted using a function written by Allan Zelener (YAD2K: Yet Another Darknet 2 Keras).

<img src= "https://github.com/JeffGoodrich9791/YOLOv2_Autonomous_Vehicle_Image_Detection/blob/master/Vehicle_Bounding_Boxes.png" />

### Model

Template code is provided in the `YOLO_Autonomous_Driving_Image_Detection.ipynb` notebook file. The layers of the network were constructed using Python 3 and Keras backend in an iPyton Notebook. The input is a batch of images of shape 608px X 608px X 3 (rgb) which is run through a Deep Convolutional Neural Network (D-CNN) with a reduction factor of 32. The output is a list of bounding boxes with a shape of 19 X 19 X 425, where 425 is the flattening of 80 classes with 5 anchor boxes each. The first 5 varialbles in the vector includes the probability (Pc), bounding box x-coord (bx),bounding box y-coord (by), bounding box hieght (bh), bounding box width (bw) followed by the classes (c).  

<img src= "https://github.com/JeffGoodrich9791/YOLOv2_Autonomous_Vehicle_Image_Detection/blob/master/Encoding_DeepCNN.png" />

For each of the 5 bounding boxes in each of the 19 X 19 cells, the elementwise product is computed to get a probability that the box contains a each of the 80 classes trained into the model. This produces a "score" for each cell as they scanned across the image. A threshold value is used to filter the scores so that only the scores above the threshold are significant. The threshold value used in the model was 0.6. 

After filtering by thresholding over the classes scores, you still end up a lot of overlapping boxes. A second filter for selecting the right boxes is called non-maximum suppression (NMS). Non-maximum supression uses Intersection over Union (IoU) to select the highest probability out of the remaining bounding boxes. 


<img src= "https://github.com/JeffGoodrich9791/YOLOv2_Autonomous_Vehicle_Image_Detection/blob/master/NMS.png" />

The output of yolo_model is a (m, 19, 19, 5, 85) tensor that needs to pass through non-trivial processing and conversion. The following cell does that for you.

### Run

The output of yolo_model is a (m, 19, 19, 5, 85) tensor that needs to pass through non-trivial processing and conversion. The following cell does executes this. 

> yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

The yolo_outputs produced all of the predicted boxes of yolo_model in the correct format. Now the the ouptuts need to be filtered using this command. 

> scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)

The prediction for the new image is then run through the following code: 

>  Preprocess your image
    image, image_data = preprocess_image("images/" + image_file, model_image_size = (608, 608))

    
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict = {yolo_model.input: image_data,    K.learning_phase():0})
    ### END CODE HERE ###

    # Print predictions info
    print('Found {} boxes for {}'.format(len(out_boxes), image_file))
    # Generate colors for drawing bounding boxes.
    colors = generate_colors(class_names)
    # Draw bounding boxes on the image file
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    # Save the predicted bounding box on the image
    image.save(os.path.join("out", image_file), quality=90)
    # Display the results in the notebook
    output_image = scipy.misc.imread(os.path.join("out", image_file))
    imshow(output_image)
    
    return out_scores, out_boxes, out_classes


### Results

> out_scores, out_boxes, out_classes = predict(sess, "test.jpg")

<img src= "https://github.com/JeffGoodrich9791/YOLOv2_Autonomous_Vehicle_Image_Detection/blob/master/Bounding_Box_Output.png" /> 

