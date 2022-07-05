import streamlit as st;
import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
from sklearn.tree import DecisionTreeClassifier, plot_tree;
import util.upload_file as upload;

st.write("""
# Linear Regression
> ### Upload a File""");

file = st.file_uploader("Choose File", type=['json', 'csv', 'xlsx', 'xls']);

if (file is not None): 
    data = upload.upload_file(file);

    st.write("> ### File's Content")
    st.dataframe(data);

    st.write("""
    > ### Parametrization
    Choose the variables to use for the decision tree
    """);

    clasification = st.selectbox("Please choose the column for clasification", data.keys());

    
