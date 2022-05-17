import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Iris Flower Prediction App
This app predicts the **Iris flower** type!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('Please input the characteristics of flower:-')
st.write(df)

iris = pd.read_csv('https://raw.githubusercontent.com/susanlum/AAA-Coursework/main/IRIS.csv')
X = iris.drop('species', axis = 1)
Y = iris.species

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
# st.write(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])

iris_list = [[0,'Iris-setosa'],[1,'Iris-versicolor'],[2,'Iris-virginica']]
# st.dataframe (iris_list, columns = ['Class_Labels', 'Iris_Type'])

df = pd.DataFrame (iris_list, columns = ['Class_Labels', 'Iris_Type'])
print(df)

st.image("https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Machine+Learning+R/iris-machinelearning.png")

st.subheader('Prediction')
#st.write(iris.target_names[prediction])
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)

st.snow()
