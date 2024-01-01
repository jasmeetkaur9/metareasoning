import collections
import time
import random
import gym
from envs.metaenv import MetaWorldEnv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.io import write_image as sv

def plot_graph(data, label, x_t, y_t, title, d = 1):
    if d == 1:
        data_arr = np.array([np.array(l) for l in data], dtype=object)
        num = data_arr.shape[0]
        x = np.array(range(data_arr.shape[1]))
        fig = go.Figure()
        for i in range(num):
            fig.add_trace(go.Scatter(x = x, y = data_arr[i], name = label[i], mode='lines+markers'))
        fig.update_layout(
        xaxis_title=x_t,
        yaxis_title=y_t,
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=1200, height=800
        )
        sv(fig, file = "plots/"+title+".png", format = 'png')
    else :
        x = np.array(range(len(data)))
        y = np.array(data)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x = x, y = y, name = label, mode='lines+markers'))
        fig.update_layout(
        xaxis_title=x_t,
        yaxis_title=y_t,
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=1200, height=800
        )
        sv(fig, file = "plots/"+title+".png", format = 'png')
