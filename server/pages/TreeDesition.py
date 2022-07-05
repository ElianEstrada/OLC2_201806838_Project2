from sklearn import preprocessing
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

    y_val = data[clasification];
    data = data.drop([clasification], axis=1);

    x_val = [];
    label_encoder = preprocessing.LabelEncoder();
    labels = data.head();
    values = labels.columns;

    for val in values :
        list_value = list(data[val]);
        transformation = label_encoder.fit_transform(list_value);
        x_val.append(transformation);

    features = list(zip(*x_val));
    label = label_encoder.fit_transform(y_val);

    clf = DecisionTreeClassifier().fit(features, label);
    fig = plt.figure();
    plt.style.use("bmh");
    plot_tree(clf, filled=True);
    plt.title("Tree Decition");

    if st.button("Clasifier"): 
        st.pyplot(fig);