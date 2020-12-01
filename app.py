import math

from flask import Flask, render_template, request, redirect, url_for, json
import getClass
import os,keras
import cv2
import numpy as np
from keras.preprocessing import image
import tensorflow as tf
from PIL import Image
import base64
import io
from tensorflow.keras.models import load_model

app = Flask(__name__)
EN_model=keras.models.load_model(os.getcwd()+"/Models/EN/EN.h5")
DN_model=keras.models.load_model(os.getcwd()+"/Models/DN/DN.h5")
resnet_model=DN_model#keras.models.load_model(os.getcwd()+"/Models/DN/DN.h5")

diseaseToClassMap={}
diseaseToClassMap['cardiomegaly']=0
diseaseToClassMap['consolidation']=1
diseaseToClassMap['pneumothorax']=2

#Function to get the best class from the model output
def getBestClass(d):
    arr=[(d['cardiomegaly'],'cardiomegaly'),(d['consolidation'],'consolidation'),(d['pneumothorax'],'pneumothorax')]
    arr.sort(reverse=True)
    return arr[0][1]

#Function to generate GradCAM image for the model
def GradCAM(IMG_PATH, MODEL_PATH, MODEL_NAME,CLASS):
    IMAGE_PATH = IMG_PATH
    LAYER_NAME=''
    if (MODEL_NAME == 'densenet'):
        LAYER_NAME = 'conv5_block16_2_conv'

    if (MODEL_NAME == 'resnet'):
        LAYER_NAME = 'conv5_block16_2_conv'

    if (MODEL_NAME == 'efficientnet'):
        LAYER_NAME = 'top_conv'

    # set this as the actual class number
    CAT_CLASS_INDEX = CLASS

    img = tf.keras.preprocessing.image.load_img(IMAGE_PATH, target_size=(224, 224))
    img_tensor = image.img_to_array(
        img)  # (height, width, channels)        # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.
    img = img_tensor

    # Load initial model
    #model = load_model(MODEL_PATH)
    model=MODEL_PATH
    # Create a graph that outputs target convolution and output
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(LAYER_NAME).output, model.output])

    # Get the score for target class
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.array([img]))
        loss = predictions[:, CAT_CLASS_INDEX]

    # Extract filters and gradients
    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]

    # Average gradients spatially
    weights = tf.reduce_mean(grads, axis=(0, 1))

    # Build a ponderated map of filters according to gradients importance
    cam = np.ones(output.shape[0:2], dtype=np.float32)

    for index, w in enumerate(weights):
        cam += w * output[:, :, index]

    # Heatmap visualization
    cam = cv2.resize(cam.numpy(), (224, 224))
    cam = np.maximum(cam, 0)
    heatmap = (cam - cam.min()) / (cam.max() - cam.min())
    cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    img *= 255.
    output_image = cv2.addWeighted(cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2BGR), 0.7, cam, 1, 0)
    #path_to_put_gradcam=os.getcwd()+'/static/GRADCAM.jpg'
    cv2.imwrite('./GRADCAM.jpg', output_image)

@app.route('/')
def hello_world():
    return render_template('firstpage_2.html')

@app.route('/data',methods=['GET','POST'])
def receiveData():
    # try:
        if request.method == 'POST':
            uploaded_file = request.files['filename']
            #print(request.form)
            chosen_model=request.form['website']
            if uploaded_file.filename != '':
                uploaded_file.save(os.getcwd()+'/images/'+uploaded_file.filename)
            captcha_response=request.form['g-recaptcha-response']
            if(captcha_response==None or captcha_response==''):
                return render_template('Invalid_Profile.html')
            image_path=os.getcwd()+"/images/"+uploaded_file.filename
            print(image_path,'\n')
            if(True):#chosen_model=='efficientnet'):
                predictedClass=getClass.load_image(image_path,EN_model)
                bestClass = getBestClass(predictedClass)
                GradCAM(image_path, EN_model, 'efficientnet', diseaseToClassMap[bestClass])
            elif(chosen_model=='densenet' or chosen_model=='resnet'):
                predictedClass=getClass.load_image(image_path,DN_model)
                bestClass = getBestClass(predictedClass)
                GradCAM(image_path, DN_model, 'densenet', diseaseToClassMap[bestClass])
            else:
                predictedClass = getClass.load_image(image_path, resnet_model)
                bestClass = getBestClass(predictedClass)
                GradCAM(image_path, resnet_model, 'resnet', diseaseToClassMap[bestClass])
            im = Image.open("GRADCAM.jpg")
            data = io.BytesIO()
            im.save(data, "JPEG")
            encoded_img_data = base64.b64encode(data.getvalue())
            # return render_template("image_tes.html", img_data=encoded_img_data.decode('utf-8'))
            temp2D=[['tasks','hours'],['cardiomegaly',math.ceil(predictedClass['cardiomegaly']*100)],
                    ['consolidation',math.ceil(predictedClass['consolidation']*100)],
                    ['pneumothorax',math.ceil(predictedClass['pneumothorax']*100)]]
            tempo=json.dumps(temp2D)
            return render_template('simplpc-4.html',
                                   cardiomegaly=predictedClass['cardiomegaly'],
                                   consolidation=predictedClass['consolidation'],
                                   pneumothorax=predictedClass['pneumothorax'],
                                   gradcam=encoded_img_data.decode('utf-8'),
                                   object=tempo
                                   )
    # except Exception as e:
    #     return e

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=3000)

#kill -9 $(ps -A | grep python | awk '{print $1}')