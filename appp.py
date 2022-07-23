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
global graph, model


app = Flask(__name__, template_folder='Template')
model = load_model('model.h5')

@app.route('/', methods=['GET'])
def index_view():
    return render_template("index.html")


@app.route('/submit', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        image = request.files['file']
        basepath = os.path.dirname(__file__)
        image_path = os.path.join(basepath, secure_filename(image.filename))
        image.save(image_path)
        img = load_img(image_path, target_size=(300, 300))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        image_tensor = np.vstack([x])
        classes = model.predict(image_tensor)
        print(classes * 10)
        if classes[0][0] > 50:
            response = 'This image is a cat'
        else:
            classes[0][1] > 50
            response = 'This image is a dog'
    return render_template("index.html", prediction=response, image=image)


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.debug = True
    app.run(host='0.0.0.0', port=port)



