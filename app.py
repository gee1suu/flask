from flask import Flask, Response, request, jsonify
from flask_cors import CORS
from tensorflow import keras
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

@app.route("/", methods=['GET', 'POST'])
def image():
    new_model = keras.models.load_model('./model.h5')

    classes = ['1', '2', '3', '4']
    data = request.get_data()

    img = Image.open(io.BytesIO(data))
    img = img.convert('RGB')
    img = img.resize([224, 224])
    img = keras.preprocessing.image.img_to_array(img)
    img = img.reshape(1, 224, 224, 3)
    img = img.astype('float32')
    result = new_model.predict(img)
    
    dict_result = {}
    for i in range(4):
        dict_result[result[0][i]] = classes[i]
    res = result[0]
    res.sort()
    res = res[::-1]
    prob = res[:3]

    return jsonify(dict_result[prob[0]])

app.run(host='0.0.0.0')