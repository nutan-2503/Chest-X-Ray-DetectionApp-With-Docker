from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

import keras
import tensorflow as tf
import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt
import numpy as np
import os

from pre_process import path_to_tensor, getting_two_layer_weights, CAM_func

app = Flask(__name__)
app.secret_key='nhotwani'

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method=='POST':
        username = request.form['username']
        password = request.form['password']

        if(username=="nhotwani@example.com" and password=="nhotwani@2021"):
            print("Logged in successfully!...")
            return redirect(url_for('upload'))

        else:
            print("Credentials incorrect!...")
            return redirect(url_for('login'))
            

    return render_template("login.html")


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if(request.method=='POST'):
        return redirect(url_for('result'))
        
    return render_template("upload.html")


@app.route('/result', methods=['GET', 'POST'])
def result():
    if(request.method=='POST'):
        # If file not uploaded successfully
        if 'file' not in request.files:
            print("Image not uploaded :(")
            return render_template("upload.html")
        
        img_file = request.files['file']

        # If file is empty
        if img_file.filename=="":
            print("Image not uploaded :(")
            return render_template("upload.html")

        if img_file:
            passed = False
            try:
                filename = secure_filename(img_file.filename)
                filepath = os.path.join("static/uploads/", filename)
                img_file.save(filepath)
                passed = True
            
            except Exception:
                passed = False
            
            if(passed):
                # Processing for GRAD-CAM visualisation
                image_to_predict = path_to_tensor(filepath).astype('float32')/255.0
                # Getting the prediction
                prediction = ensemble_model.predict(image_to_predict)
                answer = {0:"NORMAL", 1:"COVID-19", 2:"BACTERIAL PNEUMONIA", 3:"VIRAL PNEUMONIA"}
                prediction_final = "Normal: " + str(np.round(prediction[0][0]*100, decimals = 2)) + "%" + \
                   " | COVID-19: " + str(np.round(prediction[0][1]*100, decimals = 2)) + "%" + \
                   " | Bacterial Pneumonia: " + str(np.round(prediction[0][2]*100, decimals = 2)) + "%" + \
                   " | Viral Pneumonia: " + str(np.round(prediction[0][3]*100, decimals = 2)) + "%"

                # Getting the weights of last activation and last dense for localization
                ensembled_model, all_amp_layer_weights = getting_two_layer_weights()
                # Saving GRAD-CAM image
                plt.ioff()
                fig = plt.figure(figsize = (5, 7))
                plt.imshow(image_to_predict.squeeze())
                plt.xticks([])
                plt.yticks([])
                plt.savefig("static/results/image1.jpg", bbox_inches='tight', pad_inches=0.1)
                plt.close(fig)

                fig = plt.figure(figsize=(5, 7))
                CAM, pred = CAM_func(filepath, ensembled_model, all_amp_layer_weights)
                CAM = (CAM - CAM.min()) / (CAM.max() - CAM.min())
                plt.imshow(image_to_predict.squeeze(), vmin=0, vmax=255)
                plt.imshow(CAM, cmap = "jet", alpha = 0.2, interpolation='nearest', vmin=0, vmax=1)
                plt.xticks([])
                plt.yticks([])
                plt.savefig("static/results/image2.jpg", bbox_inches='tight', pad_inches=0.1)
                plt.close(fig)

                return render_template("result.html", pred=answer[np.argmax(prediction)], final_result=prediction_final)
            
            else:
                print("Image not uploaded :(")
                return render_template("upload.html")
    
    return render_template("upload.html")


if __name__ == '__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    ensemble_model = tf.keras.models.load_model("./weights/ensemble_model.hdf5", compile=False)
    app.run(host="127.0.0.1", port="5000")