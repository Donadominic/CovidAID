from flask import Flask, render_template, request
import keras
from keras.models import load_model
from keras.preprocessing import image
from werkzeug.utils import secure_filename
import numpy as np
import sys
import os



app=Flask(__name__)


MODEL_PATH = "Model/best_model_6.hdf5"
model = load_model(MODEL_PATH)
print("model is loaded")
categories = ['Bacterial', 'Covid', 'Normal', 'TB', 'Viral']



@app.route('/')
def index():
	return render_template("index.html")
	

@app.route("/prediction", methods=["POST"])	
def upload():
	
	img=request.files['img']
	
	basepath=os.path.dirname(__file__)
	file_path=os.path.join(basepath, 'uploads', secure_filename(img.filename))
	img.save(file_path)
	
	
	
	test_image = image.load_img(file_path, target_size=(224, 224))

	test_image = image.img_to_array(test_image)
	
	test_image=np.array([test_image], dtype=np.float16) / 255.0 
	
	
	result = model.predict(test_image)
	
	result=categories[np.argmax(result)]
	
	if result=='Bacterial': 
		return render_template("prediction.html", data="Bacteral Pneumonia")
	elif result=='Viral': 
		return render_template("prediction.html", data="Viral Pneumonia")
	elif result=='TB': 
		return render_template("prediction.html", data="Tuber Culosis")
	elif result=='Covid':
		return render_template("prediction.html", data="Covid Positive")
	else :
		return render_template("prediction.html", data="Covid Negative")
	
	
	#return render_template("prediction.html", data=result)
	
	

if __name__== "__main__":
	app.run()
	