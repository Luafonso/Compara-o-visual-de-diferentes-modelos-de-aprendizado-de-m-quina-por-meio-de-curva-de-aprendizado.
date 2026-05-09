import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt 
import plotly.express as px
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score


def primeiro_modelo(df):

    X = df[['Sex', 'Age', 'Pclass', 'Fare']]
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


    clf = tree.DecisionTreeClassifier(max_depth=3, random_state=42)
    clf = clf.fit(X_train, y_train)
    predicoes = clf.predict(X_test)

    acc = accuracy_score(y_test, predicoes)
    return predicoes, acc, y_test


def segundo_modelo(df):

    X = df[['Sex', 'Age', 'Pclass', 'Fare']]
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


    clf = tree.DecisionTreeClassifier(max_depth=15, random_state=42)
    clf = clf.fit(X_train, y_train)
    predicoes = clf.predict(X_test)

    acc = accuracy_score(y_test, predicoes)
    return predicoes, acc, y_test


st.set_page_config(
    page_icon=':rocket:',
    page_title='Comparador de modelos'
)

st.title(':rocket: Comparador de modelos de Machine Learning')

# Fazendo com que tenha upload do csv
uploaded_files = st.file_uploader(
    label='Upload data', type=['csv']
)


if uploaded_files:
    # Lendo os arquivos 
    df = pd.read_csv(uploaded_files)
    df = df.dropna()
    st.dataframe(df)
    df['Sex'] = df['Sex'].map({
        'male': 0,
        'female': 1
    }) 
   
    # Fazer o treinamento
    pred1, acc1, y_test = primeiro_modelo(df)

    pred2, acc2, y_test = segundo_modelo(df)

    sobreviveu_modelo1 = sum(pred1)

    sobreviveu_modelo2 = sum(pred2)

    sobreviveu_original = sum(y_test)

    acc1 = f'{(acc1 * 100):.2f}%'
    acc2 = f'{(acc2 * 100):.2f}%'

    comparacao = pd.DataFrame({
        'Modelo': [
            'Real',
            'Árvore rasa',
            'Árvore profunda'
        ],

        'Sobreviventes': [
            sobreviveu_original,
            sobreviveu_modelo1,
            sobreviveu_modelo2
        ],

        'Acurácia': [
            'N/A',
            acc1,
            acc2
        ]
    })


    # Comparações
    st.subheader('Sobreviveu ou morreu no dataset')
    st.write((df['Survived'] == 1).value_counts())

    st.subheader('Comparação de sobreviventes')
    st.dataframe(comparacao)

    fig = px.bar(
        comparacao,
        x='Modelo',
        y='Sobreviventes',
        color='Modelo',
        text='Sobreviventes',
        title='Comparação de Sobreviventes',
        template='plotly_dark'
    )

    st.plotly_chart(fig)