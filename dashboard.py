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

def comparar_curvas_aprendizado(modelo1, modelo2, X, y):
    # 1. Calcula a curva para o Modelo 1 (Árvore Rasa)
    tam_treino1, scores_t1, scores_v1 = learning_curve(
        modelo1, X, y, cv=5, scoring='accuracy', n_jobs=-1, random_state=42
    )
    # 2. Calcula a curva para o Modelo 2 (Árvore Profunda)
    tam_treino2, scores_t2, scores_v2 = learning_curve(
        modelo2, X, y, cv=5, scoring='accuracy', n_jobs=-1, random_state=42
    )
    
    # Tira a média dos resultados
    t1_medio, v1_medio = scores_t1.mean(axis=1), scores_v1.mean(axis=1)
    t2_medio, v2_medio = scores_t2.mean(axis=1), scores_v2.mean(axis=1)
    
    # Criando o gráfico comparativo lado a lado
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot Modelo 1
    ax1.plot(tam_treino1, t1_medio, 'o-', color="red", label="Treino (Rasa)")
    ax1.plot(tam_treino1, v1_medio, 'o-', color="green", label="Validação (Rasa)")
    ax1.set_title("Aprendizado: Árvore Rasa")
    ax1.set_xlabel("Amostras de Treino")
    ax1.set_ylabel("Acurácia")
    ax1.legend()
    ax1.grid(True)
    
    # Plot Modelo 2
    ax2.plot(tam_treino2, t2_medio, 'o--', color="red", label="Treino (Profunda)")
    ax2.plot(tam_treino2, v2_medio, 'o--', color="green", label="Validação (Profunda)")
    ax2.set_title("Aprendizado: Árvore Profunda")
    ax2.set_xlabel("Amostras de Treino")
    ax2.legend()
    ax2.grid(True)
    
    return fig


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

    # 1. Chame as funções atualizadas (repare na quarta variável recebendo o modelo)
    predicao_m1, acc_m1, target_teste, modelo_raso = primeiro_modelo(df)
    predicao_m2, acc_m2, target_teste, modelo_profundo = segundo_modelo(df)

    # ... seu código de montagem da tabela e do gráfico de barras ...

    # 2. Seção de comparação de aprendizado
    st.markdown("---")
    st.subheader("📈 Comparação Direta do Aprendizado")
    st.write("Analise o comportamento de treino vs. validação de ambos os modelos lado a lado:")
    
    # Separando os dados necessários para o teste da curva
    X = df[['Sex', 'Age', 'Pclass', 'Fare']]
    y = df['Survived']
    
    # Gerando e exibindo o gráfico comparativo
    fig_comparacao = comparar_curvas_aprendizado(modelo_raso, modelo_profundo, X, y)
    st.pyplot(fig_comparacao)

