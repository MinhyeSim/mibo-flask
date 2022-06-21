import os
import sys
from model.cabbage.cabbage import Solution
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from flask import Flask, render_template,request
import icecream as ic


app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('cabbage.html')


@app.route("/cabbage", methods=["post"])
def cabbage():
    avg_temp = request.form['avg_temp']
    min_temp = request.form['min_temp']
    max_temp = request.form['max_temp']
    rain_fall = request.form['rain_fall']
    ic(f'{avg_temp}, {min_temp}, {max_temp},{rain_fall}')
    render_params = {}
    render_params['result'] = result
    return render_template('index.html', **render_params)
    

if __name__ == '__main__':
    print(f'Started Server')
    app.run() 