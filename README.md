# Chest-X-Ray-DetectionApp-With-Docker
A FLASK Application to detect whether the given Chest-X-Ray is Normal, COVID infected or Pneumonia infected (Bacterial and Viral). The Application takes input a chest x ray and predicts the result using Deep Learning models in the backend and displays the output with an additional feature of GRAD-CAM visualisation. <br/>
**We can even launch the application using Docker in any system of our choice.**<br/>
Technologies Used:
- Deep Learning
- Python
- Flask
- Docker
- HTML/CSS

# Deep Learning Model
The Application first builds a Deep Learning model to predict the deformity in the chest-x-ray. The steps followed in building a model of our accuracy: 
- MobileNet model: I first tried to build a pre-trained mobilenet model with test accuracy of 76%.
- Inception model: Because of the low accuracy of mobilenet, I built the inception model with test accuracy of 66%.
- Ensemble model: Both the stated accuracies were not sufficient, which led be built an ensembled model, combination of both mobilenet and inception, which increased the accuracy to 80%. I was able to achieve a model that could predict diformity in chets-x-ray with accuracy of 80%.
- GRAD-CAM visualisation: GRAD-CAM visualisation is an important tool in Computer Vision that helps to visualise the image prediction by the machine.

# Flask API
Flask is a Python-based framework that provides us the ease of dealing with microservices. My application makes use of the framework and builds a REST service that interacts with user and builds an application to detect deformity in Chest-X-Ray. It builds several endpoints for user-interaction.
- Login: Takes in the credentials from the user and proceeds to next page if login successful.
- Upload: Prompts the user to upload the chest-x-ray.
- Result: Displays the result with GRAD-CAM image for proper user understanding.


