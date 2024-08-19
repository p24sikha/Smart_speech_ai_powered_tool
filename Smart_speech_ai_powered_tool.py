import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import speech_recognition as sr
import numpy as np
import cv2
from transformers import pipeline
import random

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

recognizer = sr.Recognizer()

nlp = pipeline("sentiment-analysis")

devices = {
    "living_room_light": False,
    "kitchen_light": False,
    "bedroom_light": False,
    "thermostat": 20,
}

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("AI-Powered Virtual Assistant for SmartHome Solutions"), className="mb-4")
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='gesture-recognition', figure=go.Figure()),
            dbc.Button("Capture Gesture", id="capture-gesture", color="primary", className="mt-2")
        ], width=6),
        dbc.Col([
            dbc.Button("Start Speech Recognition", id="start-speech", color="success", className="mb-2"),
            html.Div(id="speech-output"),
            html.Div(id="nlp-output"),
        ], width=6)
    ]),
    dbc.Row([
        dbc.Col([
            html.H3("Smart Home Devices", className="mt-4"),
            html.Div(id="device-status")
        ])
    ])
])

@app.callback(
    [Output("speech-output", "children"),
     Output("nlp-output", "children")],
    Input("start-speech", "n_clicks"),
    prevent_initial_call=True
)
def recognize_speech(n_clicks):
    with sr.Microphone() as source:
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        sentiment = nlp(text)[0]
        return f"You said: {text}", f"Sentiment: {sentiment['label']} ({sentiment['score']:.2f})"
    except sr.UnknownValueError:
        return "Could not understand audio", ""
    except sr.RequestError:
        return "Could not request results", ""

@app.callback(
    Output("gesture-recognition", "figure"),
    Input("capture-gesture", "n_clicks"),
    prevent_initial_call=True
)
def capture_gesture(n_clicks):
    gesture_data = np.random.rand(10, 10)
    return go.Figure(data=go.Heatmap(z=gesture_data))

@app.callback(
    Output("device-status", "children"),
    [Input("speech-output", "children"),
     Input("gesture-recognition", "figure")],
    prevent_initial_call=True
)
def control_devices(speech, gesture):
    global devices
    
    if speech and "light" in speech.lower():
        devices["living_room_light"] = not devices["living_room_light"]
    elif gesture:
        devices["thermostat"] += 1

    status = []
    for device, state in devices.items():
        if isinstance(state, bool):
            status.append(dbc.Badge(f"{device}: {'ON' if state else 'OFF'}", 
                                    color="success" if state else "danger",
                                    className="me-1"))
        else:
            status.append(dbc.Badge(f"{device}: {state}°C", color="info", className="me-1"))
    
    return status

if __name__ == '__main__':
    app.run_server(debug=True)