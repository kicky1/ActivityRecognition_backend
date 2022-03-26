from operator import methodcaller
from flask import Flask
from flask import jsonify, request
from datetime import datetime
import json
import base64
from base64 import b64encode
from flask_cors import CORS, cross_origin
from ai import create_directory
import tensorflow as tf
from tensorflow.python.keras.models import load_model
from glob import glob
import cv2
from collections import Counter


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


model = load_model('C:/Users/kwicki/Desktop/Magisterka/WebApp/backend_activityclassification/model.hdf5')

@app.route('/api/query', methods = ['POST'])
def get_query_from_react():
    data = request.get_json()
    path = data['data']
    directory = 'C:/Users/kwicki/Desktop/Magisterka/test_data/prediction'

    try:
        print(path)
        directory = create_directory(path, directory)
        print(path)
        pred_li = []
        images = glob(directory + '/*.png')

        for i in images:
            image = cv2.imread(i)
            pred = model.predict(image.reshape((1, 128, 128, 3)))
            
            y_classes = pred.argmax(axis=-1)
            pred_li.append(y_classes[0])    
            
        most_common,num_most_common = Counter(pred_li).most_common(1)[0]

        if(most_common == 0):
            response = f'Class 0 \n Number of beats of class 0 are {str(num_most_common)} out of {str(len(images))}'
            return response

        elif(most_common == 1):
            response = f'Class 1 \n Number of beats of class 1 are {str(num_most_common)} out of {str(len(images))}'
            return response
                
        elif(most_common == 2):
            data = { 
                "mostCommon": num_most_common,
                "numImages": len(images),
                "activity": 'siedzenie'
                }
            response=json.dumps(data)
            
            return response
    except:
        return 'error'




@app.route('/')
def main():
    return "Server is running"




if __name__ == "__main__":
    app.run()
