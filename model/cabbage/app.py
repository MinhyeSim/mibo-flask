import os
import sys
from model.cabbage.cabbage import Solution

from model.calc_model import CalcModel
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from flask import Flask, render_template,request

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/cabbage", methods=["post"])
def cabbage():   

    result = cabbage.processing()
    return render_template('index.html', **render_params)
    

if __name__ == '__main__':
    print(f'Started Server')
    app.run() 