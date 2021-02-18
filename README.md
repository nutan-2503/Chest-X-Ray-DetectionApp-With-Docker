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
- Login: Takes in the credentials from the user and proceeds to next page if login successful.<br/> <br/>![login](https://user-images.githubusercontent.com/60135434/108323199-61163d00-71ec-11eb-9d90-d3e35288d4e2.png)
- Upload: Prompts the user to upload the chest-x-ray. <br/><br/>![upload](https://user-images.githubusercontent.com/60135434/108323556-d5e97700-71ec-11eb-9717-83c22f5b4e8d.png)
- Result: Displays the result with GRAD-CAM image for proper user understanding.<br/><br/>![result](https://user-images.githubusercontent.com/60135434/108323755-0c26f680-71ed-11eb-8854-8e0aff5e4c7d.png)

# Steps to run application:
- Clone the Application:<br/> 
````
git clone https://github.com/nutan-2503/Chest-X-Ray-DetectionApp-With-Docker.git
````
- **To run the application in python:**<br/>
````
python app.py
````
Run http://127.0.0.0/5000/ to test the application

- **Using Docker:**<br/> a. Build the docker image using:
 ````
 docker build --tag=chest-x-ray:latest .
 ````
 b. Run the app using 
 ````
 docker run -d -p 5000:5000 chest-x-ray:latest
 ````
 Run http://127.0.0.1/5000/ to test the application
