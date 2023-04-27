import os
import io
import json
import numpy as np
from PIL import Image
from bottle import route, run, request
import tensorflow as tf
from tensorflow import keras
model_path = "E:/Plants/plantsclassifier.h5"
model = keras.models.load_model(model_path)
@route('/predict', method='POST')
def predict():
    # get the image from the request
    img = request.files.get('image').file.read()
    img = Image.open(io.BytesIO(img))

    # preprocess the image
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = img.reshape((1,) + img.shape)

    # make the prediction
    pred = model.predict(img)
    pred_class = np.argmax(pred)
    classify={'critically_endangered':0,'endangered':1,'endemic':2,'near_threatened':3,'rare':4,'vulnerable':5}
    # Print the predicted class label
    def get_key(pred_class):
        for key, value in classify.items():
            if pred_class == value:
                return key
 
        return "key doesn't exist"

    # return the prediction result as JSON
    return json.dumps({'class': get_key(pred_class)})

@route('/')
def upload_form():
    return '''
<!DOCTYPE html>
<html>
<head>
	<title>Plant Classifier</title>
	<meta charset="utf-8">
	<meta name="viewport" content="width=device-width, initial-scale=1">
	<style>
		body {
			font-family: Arial, Helvetica, sans-serif;
			background-color: #f2f2f2;
			margin: 0;
			padding: 0;
		}

		.container {
			background-color: #ffffff;
			margin: 50px auto;
			padding: 20px;
			border-radius: 10px;
			box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
			max-width: 600px;
		}

		h1 {
			font-size: 32px;
			color: #333333;
			text-align: center;
		}

		form {
			margin-top: 30px;
			text-align: center;
		}

		input[type="file"] {
			margin: 20px 0;
			padding: 10px;
			border: 2px dashed #cccccc;
			border-radius: 5px;
			background-color: #f9f9f9;
			color: #333333;
			font-size: 16px;
			font-weight: bold;
			cursor: pointer;
			width: 100%;
			max-width: 400px;
		}

		input[type="submit"] {
			padding: 10px 20px;
			border: none;
			border-radius: 5px;
			background-color: #333333;
			color: #ffffff;
			font-size: 16px;
			font-weight: bold;
			cursor: pointer;
			transition: background-color 0.3s ease-in-out;
		}

		input[type="submit"]:hover {
			background-color: #666666;
		}
	</style>
</head>
<body>
	<div class="container">
		<h1>Plant Classifier</h1>
		<form action="/predict" method="post" enctype="multipart/form-data">
			<label for="image">Select an image:</label><br>
			<input type="file" name="image" id="image"><br>
			<small>Accepted formats: JPG, JPEG, PNG.</small><br>
			<small>Max file size: 5MB.</small><br><br>
			<input type="submit" value="Classify Plant">
		</form>
	</div>
</body>
</html>

'''
if __name__ == '__main__':
    run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))


