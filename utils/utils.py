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


def plot_graph(data, label, title):
    x = np.array(range(len(data)))
    y = np.array(data)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = x, y = y, name = label, mode='lines+markers'))
    fig.update_layout(
    xaxis_title=label,
    yaxis_title='Reward',
    yaxis=dict(scaleanchor="x", scaleratio=1),
    xaxis=dict(constrain='domain'),
    width=1200, height=800
    )
    sv(fig, file = "plots/"+title+".png", format = 'png')
