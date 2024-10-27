from flask import Flask, render_template, request
from markupsafe import Markup 
import pickle
import numpy as np

app = Flask(__name__) # invoke the Flask class

@app.route("/",methods = ['GET', 'POST'])
def get_iris():
    pred_names = ['Iris Setosa', 'Iris Versicolor', 'Iris Virginica']
    pred_label = ''
    filename = '/home/lillianzhang/iris-dnn/hosted/dnn_model.pkl'
    file = open(filename, 'rb')
    
    # load model
    model = pickle.load(file)
    
    iris_measurements = np.array([])
    if request.method == 'GET':
        
        return render_template('index.html')
        
    elif request.method == 'POST':
        sl = request.form["sl"]
        sw = request.form["sw"]
        pl = request.form["pl"]
        pw = request.form["pw"]

        new_iris = [sl, sw, pl, pw]
        if all(new_iris): # if we have all inputs from form
            new_iris_float = np.array([[float(x) for x in new_iris]])
            pred = np.argmax(model.predict(new_iris_float), axis = 1)
            pred_label = pred_names[int(pred)]

        pred_sentence = Markup(f"Your iris with the below measurements<br><br>Sepal length {sl}cm<br>\
        Sepal width {sw}cm<br>Petal length {pl}cm<br>Petal width {pw}cm<br><br>is a {pred_label}")
        return render_template('index.html', prediction = pred_sentence)


# if __name__ == '__main__':
#     app.run() # Start the server listening for requests

