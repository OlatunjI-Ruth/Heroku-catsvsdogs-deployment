from flask import Flask, render_template, request, send_file
from PIL import Image
import io
import numpy as np
from keras.models import load_model
import re
import pathlib
import sys
import os
from keras.utils import load_img, img_to_array
from werkzeug.utils import secure_filename
sys.path.append(os.path.abspath("./model"))

'''def get_model():
    global graph, model
    model = load_model('model.h5')
    print("model loaded!")'''
global graph, model


app = Flask(__name__, template_folder='Template')
model = load_model('model.h5')

@app.route('/')
def index_view():
    return render_template("index.html")


@app.route('/submit', methods=['POST'])
def predict():
    if request.method == 'POST':
        image = request.files['my_image']
        image_path = "./media/" + image.filename
        image.save(image_path)

        img = load_img(image_path, target_size=(150, 150))
        x = img_to_array(img)
        """ x = x[:, :, 0]
        x = x.reshape(1, 150, 150, 3)"""
        x = np.expand_dims(x, axis=0)
        image_tensor = np.vstack([x])
        classes = model.predict(image_tensor)
        print(classes[0])
        if classes[0][0] > 50:
            response = 'This image is a cat'
        else:
            classes[0][1] > 70
            response = 'This image is a dog'
    return render_template("index.html", prediction=response, image=image)


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.debug = True
    app.run(host='0.0.0.0', port=port)



