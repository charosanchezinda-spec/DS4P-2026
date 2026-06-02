import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import warnings
import logging
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("balance").setLevel(logging.ERROR)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import ttest_ind, mannwhitneyu, levene
from balance.weighting_methods.rake import rake, prepare_marginal_dist_for_raking
from balance import Sample

# ==========================================
# 1. CONFIGURACIÓN DE PÁGINA Y TEXTOS
# ==========================================

st.set_page_config(page_title="Sistema de Ponderación Electoral", page_icon="📊")
st.title("📊 Sistema de Ponderación Electoral mediante Raking")
st.markdown("""
Esta es una aplicación de ponderación muestral construida 100% en Python.
Permite a los usuarios cargar sus encuestas, elegir la población target y elegir la ventana temporal para trackear la evolución de la imagen del candidato a través del tiempo.
""")
st.divider()
