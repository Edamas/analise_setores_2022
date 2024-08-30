import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import os

st.set_page_config(layout='wide', initial_sidebar_state='auto')

@st.cache_data
def carregar_dados(arquivo):
    return pd.read_excel(arquivo)


@st.cache_data
def gerar_dicionario():
    with open('dicionario_variaveis.csv', 'rt', encoding='utf-8') as file:
        dicionario_raw = [line.split('\t') for line in file.read().split('\n')]
    return pd.DataFrame(dicionario_raw[1:], columns=dicionario_raw[0])


def renomear_colunas(data, dicionario):
    colunas = []
    for coluna_data in data.columns:
        for coluna_dicionario, descricao_dicionario in zip(dicionario['VARIÁVEL'], dicionario['DESCRIÇÃO']):
            if coluna_data.strip().lower() == coluna_dicionario.strip().lower():
                colunas.append(coluna_data + '-' + descricao_dicionario)
                break
        else:
            colunas.append(coluna_data)
    data.columns = colunas
    return data


arquivo = '2022_Agregados_preliminares_por_setores_censitarios_BR.xlsx'
data = carregar_dados(arquivo)
st.dataframe(data, use_container_width=True)

# Gerar e aplicar o dicionário de descrições
dicionario = gerar_dicionario()
data = renomear_colunas(data, dicionario)

# Selecionando colunas para gráficos e análises (colunas que começam com 'v0' e a coluna de área)
colunas_disponiveis = [col for col in data.columns if col.lower().startswith('v0') or 'área' in col.lower()]

# Selecione uma coluna para o gráfico 1
coluna1 = st.sidebar.selectbox("Escolha a coluna para o Gráfico 1", colunas_disponiveis, key="coluna1")

# Selecione uma coluna para o gráfico 2
coluna2 = st.sidebar.selectbox("Escolha a coluna para o Gráfico 2", colunas_disponiveis, key="coluna2")

# Checkboxes para gráficos
histogram_check = st.sidebar.checkbox("Histogram")
boxplot_check = st.sidebar.checkbox("Box Plot")
violin_check = st.sidebar.checkbox("Violin Plot")
density_check = st.sidebar.checkbox("Density Plot")

# Gerar gráficos baseados na seleção
col1, col2 = st.columns(2)
if histogram_check:
    with col1:
        fig1 = px.histogram(data, x=coluna1, title=f"Histogram of {coluna1}")
        fig1.update_traces(opacity=0.5)
        st.plotly_chart(fig1)
    with col2:
        fig2 = px.histogram(data, x=coluna2, title=f"Histogram of {coluna2}")
        fig2.update_traces(opacity=0.5)
        st.plotly_chart(fig2)

if boxplot_check:
    with col1:
        fig1 = px.box(data, y=coluna1, title=f"Box Plot of {coluna1}")
        fig1.update_traces(opacity=0.5)
        st.plotly_chart(fig1)
    with col2:
        fig2 = px.box(data, y=coluna2, title=f"Box Plot of {coluna2}")
        fig2.update_traces(opacity=0.5)
        st.plotly_chart(fig2)

if violin_check:
    with col1:
        fig1 = px.violin(data, y=coluna1, box=True, title=f"Violin Plot of {coluna1}")
        fig1.update_traces(opacity=0.5)
        st.plotly_chart(fig1)
    with col2:
        fig2 = px.violin(data, y=coluna2, box=True, title=f"Violin Plot of {coluna2}")
        fig2.update_traces(opacity=0.5)
        st.plotly_chart(fig2)

if density_check:
    with col1:
        fig1 = px.density_contour(data, x=coluna1, title=f"Density Plot of {coluna1}")
        fig1.update_traces(opacity=0.5)
        st.plotly_chart(fig1)
    with col2:
        fig2 = px.density_contour(data, x=coluna2, title=f"Density Plot of {coluna2}")
        fig2.update_traces(opacity=0.5)
        st.plotly_chart(fig2)

# Checkboxes para análises de machine learning e estatísticas
kmeans_check = st.sidebar.checkbox("KMeans Clustering")
pca_check = st.sidebar.checkbox("PCA")
regression_check = st.sidebar.checkbox("Linear Regression")
decision_tree_check = st.sidebar.checkbox("Decision Tree Classifier")
random_forest_check = st.sidebar.checkbox("Random Forest Classifier")

# Executar as análises de Machine Learning e Estatísticas selecionadas
if kmeans_check:
    num_clusters = st.sidebar.slider('Escolha o número de clusters', min_value=2, max_value=10, value=3)
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(data[colunas_disponiveis])
    data['cluster'] = kmeans.labels_
    
    fig = px.scatter(data, x=coluna1, y=coluna2, color='cluster', title=f"KMeans Clustering com {num_clusters} Clusters",
                     opacity=0.5, size_max=5)
    fig.update_traces(marker=dict(size=5))  # Reduz o tamanho dos pontos
    st.plotly_chart(fig)

if pca_check:
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[colunas_disponiveis])

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data_scaled)

    data['pca-one'] = pca_result[:, 0]
    data['pca-two'] = pca_result[:, 1]

    fig = px.scatter(data, x='pca-one', y='pca-two', title="PCA - Primeiro e Segundo Componentes",
                     opacity=0.5, size_max=5)
    fig.update_traces(marker=dict(size=5))  # Reduz o tamanho dos pontos
    st.plotly_chart(fig)

if regression_check:
    reg = LinearRegression()
    reg.fit(data[[coluna1]], data[coluna2])

    predictions = reg.predict(data[[coluna1]])

    fig = px.scatter(data, x=coluna1, y=coluna2, title="Linear Regression",
                     opacity=0.5, size_max=5)
    fig.add_trace(go.Scatter(x=data[coluna1], y=predictions, mode='lines', name='Predicted', line=dict(width=2)))
    fig.update_traces(marker=dict(size=5))  # Reduz o tamanho dos pontos
    st.plotly_chart(fig)

    st.write(f"R^2 Score: {r2_score(data[coluna2], predictions)}")
    st.write(f"Mean Squared Error: {mean_squared_error(data[coluna2], predictions)}")

if decision_tree_check:
    tree = DecisionTreeClassifier()
    tree.fit(data[[coluna1]], data[coluna2])

    predictions = tree.predict(data[[coluna1]])

    fig = px.scatter(data, x=coluna1, y=coluna2, title="Decision Tree Classifier",
                     opacity=0.5, size_max=5)
    fig.add_trace(go.Scatter(x=data[coluna1], y=predictions, mode='markers', name='Predicted',
                             marker=dict(size=5, color='red')))
    fig.update_traces(marker=dict(size=5))  # Reduz o tamanho dos pontos
    st.plotly_chart(fig)

if random_forest_check:
    forest = RandomForestClassifier()
    forest.fit(data[[coluna1]], data[coluna2])

    predictions = forest.predict(data[[coluna1]])

    fig = px.scatter(data, x=coluna1, y=coluna2, title="Random Forest Classifier",
                     opacity=0.5, size_max=5)
    fig.add_trace(go.Scatter(x=data[coluna1], y=predictions, mode='markers', name='Predicted',
                             marker=dict(size=5, color='green')))
    fig.update_traces(marker=dict(size=5))  # Reduz o tamanho dos pontos
    st.plotly_chart(fig)