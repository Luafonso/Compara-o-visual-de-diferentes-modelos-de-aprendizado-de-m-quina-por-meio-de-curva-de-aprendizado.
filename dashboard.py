import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt 
import plotly.express as px
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score

# Ajuda a melhorar a segurança  da CPU/RAM
@st.cache_data
def primeiro_modelo(df):
    
    busca_dataframe = df[['Sex', 'Age', 'Pclass', 'Fare']]
    target = df['Survived']

    busca_dataframe_treino, busca_dataframe_teste, target_treino, target_teste = train_test_split(
        busca_dataframe,
        target,
        test_size=0.2,
        random_state=42
    )  

    arvore_decisao = tree.DecisionTreeClassifier(max_depth=3, random_state=42)
    
    arvore_decisao = arvore_decisao.fit(busca_dataframe_treino, target_treino)
    
    predicoes = arvore_decisao.predict(busca_dataframe_teste)

    acuracia = accuracy_score(target_teste, predicoes)
    
    return predicoes, acuracia, target_teste, arvore_decisao

@st.cache_data
def segundo_modelo(df):

    busca_dataframe = df[['Sex', 'Age', 'Pclass', 'Fare']]
    
    target = df['Survived']
    
    busca_dataframe_treino, busca_dataframe_teste, target_treino, target_teste = train_test_split(
        busca_dataframe,
        target,
        test_size=0.2,
        random_state=42
    )  

    arvore_decisao = tree.DecisionTreeClassifier(max_depth=15, random_state=42)
        
    arvore_decisao = arvore_decisao.fit(busca_dataframe_treino, target_treino)

    predicoes = arvore_decisao.predict(busca_dataframe_teste)

    acuracia = accuracy_score(target_teste, predicoes)

    return predicoes, acuracia, target_teste, arvore_decisao

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

    colunas_obrigatorias = {'Sex', 'Age', 'Pclass', 'Fare', 'Survived'}
    
    # Verifica se todas as colunas necessárias existem
    if not colunas_obrigatorias.issubset(df.columns):
        st.error("Erro de Segurança/Validação: O arquivo enviado não possui a estrutura correta.")
        st.stop() # Interrompe a execução com segurança

    st.dataframe(df)
    df['Sex'] = df['Sex'].map({
        'male': 0,
        'female': 1
    }) 

    # Retira o valores nulos melhorando a segurança do código e evitando que quebre o ML
    df = df.dropna()
   
    # Fazer o treinamento
    predicao_modelo1, acuracia_modelo1, target_teste, arvore_decisao = primeiro_modelo(df)

    predicao_modelo2, acuracia_modelo2, target_teste, arvore_decisao = segundo_modelo(df)

    sobreviveu_modelo1 = sum(predicao_modelo1)

    sobreviveu_modelo2 = sum(predicao_modelo2)

    sobreviveu_original = sum(target_teste)
    
    acuracia_modelo1 = f'{(acuracia_modelo1 * 100):.2f}%'
    acuracia_modelo2 = f'{(acuracia_modelo2 * 100):.2f}%'

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
            #acc1,
            acuracia_modelo1,
            
            #acc2
            acuracia_modelo2
            
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

