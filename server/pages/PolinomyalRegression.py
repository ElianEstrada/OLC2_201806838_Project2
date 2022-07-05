from sklearn.preprocessing import PolynomialFeatures
import streamlit as st;
import numpy as np;
import matplotlib.pyplot as plt;
import pandas as pd;
from sklearn.linear_model import LinearRegression;
from sklearn.metrics import mean_squared_error, r2_score;
import util.upload_file as upload;

st.write("""
# Polinomyal Regression
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

    st.write("> ### Degree");
    degree = st.radio("Choose the grade of the model", (2, 3, 4, 5), horizontal=True);

    st.write("> ### Prediction");
    pred = st.number_input("Prediction's value", None, None, 0, 1);

    #Start whit analization data
    dot_x = np.asarray(data[x_var]).reshape(-1, 1);
    dot_y = data[y_var];

    # Polinomyal Regression config
    polinomyal = PolynomialFeatures(degree=degree);
    transformation = polinomyal.fit_transform(dot_x);

    # Linear Regression config
    regression = LinearRegression();
    regression.fit(transformation, dot_y);
    prediction = regression.predict(transformation);
    r2 = r2_score(dot_y, prediction);

    #Prediction
    x_min = pred;
    x_max = pred;
    new_x = np.linspace(x_min, x_max, 1)[:, np.newaxis];
    transformation = polinomyal.fit_transform(new_x);
    value_predict = regression.predict(transformation);

    # Plot
    fig = plt.figure();
    plt.style.use("bmh");
    plt.scatter(dot_x, dot_y, color="red");
    plt.plot(dot_x, prediction, color="blue");
    plt.title(f"Polinomyal Regression with degree={degree}");
    plt.ylabel(y_var);
    plt.xlabel(x_var);

    #Show image

    if st.button('Analyze Inputs'): 

        # Calcules
        zero = round(regression.intercept_, 4);

        st.write("### Tendency Function");
        count = degree;
        func = "f(x)=";
        while count > 0:
            func += f"{'' if regression.coef_[count] < 0 else '+'}{round(float(regression.coef_[count]), 8)}x{f'^{count}' if count != 1 else ''}"
            count -= 1;

        func += f"{'' if zero < 0 else '+'}{zero}"
        st.latex(func);

        st.write("### Graph");
        st.pyplot(fig);

        st.write("### Graph's Information");

        column1, column2 = st.columns(2);

        with column1:
            st.write("Coeficent of Function");
            count = 1;
            for item in regression.coef_[1:]:
                st.write(f"- **x^{count}:** {item}");
                count += 1;
        
        with column2:
            column2.metric("R²", round(r2, 4));   

        st.metric("Intesection", zero, "-" if zero < 0 else "+");
        st.metric("Mean Square Error", mean_squared_error(dot_y, prediction));

        st.subheader("Prediction");
        st.metric(f"For {pred} the value is: ", value_predict, "-" if value_predict < 0 else "+");

        st.write("### Conclusion");

        if r2 < 0.50: 
            st.write("""
            > As can be seen, both in the trend and in the separate data that we have thanks to the polinomyal regression graph, our R² is below a moderately acceptable value, which means that our data does not fit the model used in this section.
            > Therefore, it is important to emphasize that our prediction lacks accuracy, so it would not be advisable to use this data to take actions; In these cases, it is recommended to use another grade of model in order to have a better fit in the data and have a better prediction.
            """);
        elif r2 < 0.9:
            st.write("""
            > As can be seen, both in the trend and in the separate data that we have thanks to the polinomyal regression graph, our R² has an acceptable value without being very exact, which means that our data tend to be adjusted in certain points but dispersed in others according to the model used in this section.
            > Therefore, it is important to emphasize that our prediction can be reliable to a certain extent, however it is still not 100% recommended. In this case, it is recommended to use another grade of model in order to have a better fit to the data and have a better prediction.
            """);
        else: 
            st.write("""
            > As can be seen, both in the trend and in the separate data that we have thanks to the polinomyal regression graph, our R² has an acceptable value and is quite accurate, so our data fit the model used in this section quite well.
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

        # Polinomyal Regression config
        polinomyal = PolynomialFeatures(degree=degree);
        transformation = polinomyal.fit_transform(dot_x);
        
        # Linear Regression config
        regression = LinearRegression();
        regression.fit(transformation, dot_y);
        prediction = regression.predict(transformation);
        r2 = r2_score(dot_y, prediction);

        #Prediction
        x_min = pred;
        x_max = pred;
        new_x = np.linspace(x_min, x_max, 1)[:, np.newaxis];
        transformation = polinomyal.fit_transform(new_x);
        value_predict = regression.predict(transformation);

        # Plot
        fig = plt.figure(); # Create fig for streamlit
        plt.style.use("bmh"); # Apply theme for graph
        plt.scatter(dot_x, dot_y, color="red"); # draw points on graph
        plt.plot(dot_x, prediction, color="blue"); # draw line on graph
        plt.title(f"Polinomyal Regression with degree={degree}"); # Add title for graph
        plt.ylabel(y_var); # Add title of x-axes
        plt.xlabel(x_var); # Add title of y-axes
        """, language="python");
