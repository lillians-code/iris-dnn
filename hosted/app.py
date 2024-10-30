from flask import Flask, render_template, request
from markupsafe import Markup 
import numpy as np
import pickle

app = Flask(__name__)  # Invoke the Flask class

# Load the model once when the application starts
filename = 'dnn_model.pkl'
with open(filename, 'rb') as file:
    model = pickle.load(file)

@app.route("/", methods=['GET', 'POST'])
def get_iris():
    pred_names = ['Iris Setosa', 'Iris Versicolor', 'Iris Virginica']
    pred_sentence = ''

    if request.method == 'POST':
        try:
            # Retrieve form data
            sl = request.form.get("sl")
            sw = request.form.get("sw")
            pl = request.form.get("pl")
            pw = request.form.get("pw")
            
            # Check if any field is missing or empty
            if not all([sl, sw, pl, pw]):
                raise ValueError("All input fields are required.")
            
            # Convert form data to float
            sl = float(sl)
            sw = float(sw)
            pl = float(pl)
            pw = float(pw)
            
            # Validate input ranges
            if not (4.3 <= sl <= 7.9):
                raise ValueError("Sepal length must be between 4.3 cm and 7.9 cm.")
            if not (2.0 <= sw <= 4.4):
                raise ValueError("Sepal width must be between 2.0 cm and 4.4 cm.")
            if not (1.0 <= pl <= 6.9):
                raise ValueError("Petal length must be between 1.0 cm and 6.9 cm.")
            if not (0.1 <= pw <= 2.5):
                raise ValueError("Petal width must be between 0.1 cm and 2.5 cm.")
            
            # Prepare input for prediction
            new_iris = [sl, sw, pl, pw]
            new_iris_float = np.array([new_iris])
            
            # Perform prediction
            pred = np.argmax(model.predict(new_iris_float), axis=1)
            pred_label = pred_names[int(pred)]
            
            # Create prediction sentence
            pred_sentence = Markup(f"Your iris with the below measurements:<br><br>"
                                   f"Sepal length: {sl} cm<br>"
                                   f"Sepal width: {sw} cm<br>"
                                   f"Petal length: {pl} cm<br>"
                                   f"Petal width: {pw} cm<br><br>"
                                   f"is a <strong>{pred_label}</strong>.")
        
        except ValueError as ve:
            # Handle ValueError (e.g., invalid input)
            pred_sentence = Markup(f"<span class='text-danger'>Error: {str(ve)}</span>")
        except Exception as e:
            # Handle other unexpected exceptions
            pred_sentence = Markup(f"<span class='text-danger'>An unexpected error occurred: {str(e)}</span>")
        
        return render_template('index.html', prediction=pred_sentence)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)  # Start the server listening for requests
