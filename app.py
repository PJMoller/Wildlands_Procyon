from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('Dashboard.html')

@app.route('/slider')
def slider():
    return render_template('Slider.html')

if __name__ == '__main__':
    app.run(debug=True)