


### 1. Imports and class names setup ### 

import gradio
import os
import torch

from model import create_effnetb0_model
from timeit import default_timer as timer
from typing import Tuple, Dict



### 2. Model and transforms preparation ###

# Setup class names
class_names = ["pizza", "steak", "sushi"]

# Create EffNetB2 model
effnetb0, effnetb0_transforms = create_effnetb0_model(num_classes=len(class_names))

# Load saved weights to the CPU
effnetb0.load_state_dict(torch.load(f="pretrained_effnet_feature_extractor_pizza_steak_sushi.pth", map_location=torch.device("cpu"),))



### 3. Predict function ###

# Create predict function
def effnetb0_predict(img) -> Tuple[Dict, float]:
    """
    Transforms and performs a prediction on img and returns prediction and time taken.
    """
    # Start the timer
    start_time = timer()
    
    # Transform the target image and add a batch dimension
    img = effnetb0_transforms(img).unsqueeze(0)
    
    # Put model into evaluation mode and turn on inference mode
    effnetb0.eval()
    with torch.inference_mode():
        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
        pred_probs = torch.softmax(effnetb0(img), dim=1)

    # Initialize an empty dictionary
    pred_labels_and_probs = {}

    # Loop over the indices of class_names, and create a prediction label and prediction probability dictionary for each prediction class (this is the required format for Gradio's output parameter)
    for i in range(len(class_names)):
        # Use the class name as the key and the corresponding predicted probability as the value
        pred_labels_and_probs[class_names[i]] = float(pred_probs[0][i])

    
    # Calculate the prediction time
    end_time = timer()
    pred_time = round(end_time - start_time, 4)
    
    # Return the prediction dictionary and prediction time 
    return pred_labels_and_probs, pred_time



### 4. Gradio app ###

# Create title, description and article strings
title = "FoodVision Mini 🍕🥩🍣"
description = "An EfficientNetB0 feature extractor computer vision model to classify images of food as pizza, steak or sushi."
article = "Created at (https://www.learnpytorch.io/09_pytorch_model_deployment/)."

# Create examples list from "examples/" directory
example_list = [["examples/" + example] for example in os.listdir("examples")]

demo = gradio.Interface(fn=effnetb0_predict, 
                        inputs="image", 
                        outputs=[gradio.Label(num_top_classes=3, label="Predictions"), gradio.Number(label="Prediction time (s)")],
                        examples=example_list, 
                        title=title,
                        description=description,
                        article=article)

# Launch the demo!
demo.launch()
