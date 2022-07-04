from turtle import color
import streamlit as st;
import numpy as np;
import matplotlib.pyplot as plt;
import pandas as pd;
from sklearn.linear_model import LinearRegression;
from sklearn.metrics import mean_squared_error, r2_score;
import util.load_file as load;

st.title('Linear Regression');

#posible descripción de lo que es una regresión lineal

st.subheader("Load File");

file = st.file_uploader("Choose File", type=['csv', 'xls', 'xlsx', 'json']);

if (file is not None): 
    data = load.load_file(file);

    st.markdown("#### Content File");
    st.dataframe(data);

    st.subheader("Parametrization");
    st.write("""
    Choose the variables to util for analice of Linear Regression
    """);

    column1, column2 = st.columns(2);

    with column1:
        st.write("#### Independent Variable (x):");
        x_var = st.selectbox("Please choose option: ", data.keys(), key="x_variable");

    with column2:
        st.write("#### Dependent Variable (y):");
        y_var = st.selectbox("Please choose option: ", data.keys(), key="y_variable");

    st.write("##### Prediction");
    st.write("#### Graph's Color");

    color1, color2 = st.columns(2);

    with color1: 
        dot_color = st.color_picker("Choose color for dot char", "#7cc");
    with color2: 
        line_color = st.color_picker("Choose color for line char", "#123456");

    
    #Start whit analization data
    dot_x = np.asarray(data[x_var]).reshape(-1, 1);
    dot_y = data[y_var];

    # Linear Regression configuration
    regression = LinearRegression();
    regression.fit(dot_x, dot_y);
    prediction = regression.predict(dot_x);
    r2 = r2_score(dot_y, prediction);


    # Plot

    fig = plt.figure();
    plt.style.use("seaborn");
    plt.scatter(dot_x, dot_y, color=dot_color);
    plt.plot(dot_x, prediction, color=line_color);
    plt.title("Linear Regression");
    plt.ylabel(y_var);
    plt.xlabel(x_var);


    #Show image

    if st.button('Generated'): 
        st.subheader("Result");
        st.pyplot(fig);

        st.write("#### Information Graph");
        

else: 
    st.warning("Expect to load file");