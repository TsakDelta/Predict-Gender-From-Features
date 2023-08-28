# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 19:48:34 2023

@author: Dimitris Tsakatsonis

project: Predict_gender_from_shoe_size(prediction)
"""

import joblib
import gradio as gr
import os

os.chdir(r"C:\Users\kast3\OneDrive\Documents\Python Scripts")
model = joblib.load("predict_gender_from_shoe_size.pkl")

def predict_gender(height, weight, shoe_size):
    features = [[height, weight, shoe_size]]
    
    gender = model.predict(features)
    
    return "Male" if gender == 0 else "Female"

iface = gr.Interface(
    fn=predict_gender,
    inputs=[
        gr.inputs.Number(label="Height"),
        gr.inputs.Number(label="Weight"),
        gr.inputs.Number(label="Shoe Size")
        ],
    outputs=gr.outputs.Textbox(label="Predicted Gender")
    )

iface.launch()