import streamlit as st;
import numpy as np;
import matplotlib.pyplot as plt;
import pandas as pd;
from sklearn.linear_model import LinearRegression;
from sklearn.metrics import mean_squared_error, r2_score;
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
    Choose the variables to util for analice of Linear Regression
    """);

    column1, column2 = st.columns(2);

    with column1:
        st.write("#### Independent Variable (x):");
        x_var = st.selectbox("Please choose the independent variable: ", data.keys());

    with column2:
        st.write("#### Dependent Variable (y):");
        y_var = st.selectbox("Please choose the Dependent variable: ", data.keys());

    st.write("> ### Prediction");
    pred = st.number_input("Prediction's value");

    #Start whit analization data
    dot_x = np.asarray(data[x_var]).reshape(-1, 1);
    dot_y = data[y_var];

    # Linear Regression configuration
    regression = LinearRegression();
    regression.fit(dot_x, dot_y);
    prediction = regression.predict(dot_x);
    r2 = r2_score(dot_y, prediction);
    value_predict = regression.predict([[pred]]);


    # Plot
    fig = plt.figure();
    plt.style.use("bmh");
    plt.scatter(dot_x, dot_y, color="red");
    plt.plot(dot_x, prediction, color="blue");
    plt.title("Linear Regression");
    plt.ylabel(y_var);
    plt.xlabel(x_var);

    #Show image

    if st.button('Analyze Inputs'): 

        # Calcules
        slope = round(float(regression.coef_), 4);
        intersection = round(float(regression.intercept_), 4);

        st.write("### Tendency Function");
        st.latex(f"f(x)={slope}x {'+ ' if intersection>=0 else ''}{intersection}");

        st.write("### Graph");
        st.pyplot(fig);

        st.write("#### Graph's Information");

        column1, column2, column3 = st.columns(3);

        column1.metric("Slope", slope, "-" if slope < 0 else "+");
        column2.metric("Intesection", intersection, "-" if intersection < 0 else "+");

        column3.metric("R²", round(r2, 4));   
        st.metric("Mean Square Error", mean_squared_error(dot_y, prediction));

        st.subheader("Prediction");
        st.metric(f"For {pred} the value is: ", value_predict, "-" if value_predict < 0 else "+");

        st.write("### Conclusion");

        if r2 < 0.50: 
            st.write("""
            > As can be seen, both in the trend and in the separate data that we have thanks to the linear regression graph, our R² is below a moderately acceptable value, which means that our data does not fit the model used in this section.
            > Therefore, it is important to emphasize that our prediction lacks accuracy, so it would not be advisable to use this data to take actions; In these cases, it is recommended to use another type of model (such as the polynomial) in order to have a better fit in the data and have a better prediction.
            """);
        elif r2 < 0.9:
            st.write("""
            > As can be seen, both in the trend and in the separate data that we have thanks to the linear regression graph, our R² has an acceptable value without being very exact, which means that our data tend to be adjusted in certain points but dispersed in others according to the model used in this section.
            > Therefore, it is important to emphasize that our prediction can be reliable to a certain extent, however it is still not 100% recommended. In this case, it is recommended to use another type of model (such as the polynomial) in order to have a better fit to the data and have a better prediction.
            """);
        else: 
            st.write("""
            > As can be seen, both in the trend and in the separate data that we have thanks to the linear regression graph, our R² has an acceptable value and is quite accurate, so our data fit the model used in this section quite well.
            > This means that the result, such as precision, can be useful and has greater certainty when analyzing possible future actions.
            """);


    if st.button('Linear Regression Code'):
        st.code("""
        # Upload File
        file = st.file_uploader("Choose File", type=['json', 'csv', 'xlsx', 'xls']);

        # Implemented function for choose file type
        data = upload.upload_file(file);

        # Having the data we proceed to obtain the points of the regression. 
        dot_x = np.asarray(data[x_var]).reshape(-1, 1); # x_var -> is value of column
        dot_y = data[y_var]; # y_var -> is value of column

        # Linear Regression configuration
        regression = LinearRegression();
        regression.fit(dot_x, dot_y);
        prediction = regression.predict(dot_x); # Get linear regresion
        r2 = r2_score(dot_y, prediction); # Get R²
        value_predict = regression.predict([[pred]]); # Get value predict for parameter 'pred'


        # Plot
        fig = plt.figure(); # Create fig for streamlit
        plt.style.use("bmh"); # Apply theme for graph
        plt.scatter(dot_x, dot_y, color="red"); # draw points on graph
        plt.plot(dot_x, prediction, color="blue"); # draw line on graph
        plt.title("Linear Regression"); # Add title for graph
        plt.ylabel(y_var); # Add title of x-axes
        plt.xlabel(x_var); # Add title of y-axes
        """, language="python");