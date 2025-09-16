import streamlit as st
import pandas as pd
import numpy as np
import pickle
import bz2
import joblib
from catboost import CatBoostClassifier

@st.cache_resource
def carregar_modelo_selecionado(nome_do_modelo):
    """
    Carrega o modelo de machine learning com base no nome selecionado.
    Lida com diferentes formatos de arquivo (.joblib para RandomForest, .cbm para CatBoost).
    """
    if nome_do_modelo == "Random Forest":
        model_path = "models/random.joblib"
        st.write(f"Carregando modelo de: {model_path}")
        try:
            modelo = joblib.load(model_path)
            return modelo
        except Exception as e:
            st.error(f"Não foi possível carregar o modelo de '{model_path}'. Erro: {e}")
            return None
            
    elif nome_do_modelo == "Catboost":
        model_path = "models/catboost.cbm"
        st.write(f"Carregando modelo de: {model_path}")
        modelo = CatBoostClassifier() 
        modelo.load_model(model_path)
        return modelo
    
    return None

def model_page():
    st.title("💳 Sistema de Detecção de Fraudes")

    # --- Seleção do Modelo ---
    model_name = st.selectbox(
        "Selecione o modelo para a predição",
        ("Random Forest", "Catboost"),
        index=None,
        placeholder="Selecione o modelo...",
    )

    # --- Lógica Principal do App ---
    # O restante da interface só aparece DEPOIS que um modelo é selecionado e carregado.
    if model_name:
        try:
            # Carrega o modelo usando a função cacheada
            model = carregar_modelo_selecionado(model_name)
            
            # Validação para garantir que o modelo foi carregado antes de continuar
            if model is None:
                st.warning("O modelo não pôde ser carregado. Verifique os logs de erro acima.")
                st.stop()

            st.sidebar.success(f"Modelo '{model_name}' carregado com sucesso!")
            
            # --- Inputs para as Features ---
            st.subheader("📝 Insira os dados da transação")

            # Define a ordem exata das colunas, igual ao treino
            feature_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

            # Dicionário para armazenar os valores de input do usuário
            input_values = {}

            # Organizar inputs em colunas para melhor visualização
            col1, col2, col3 = st.columns(3)

            with col1:
                input_values['Time'] = st.number_input('Time', value=0.0, format="%.2f")

            with col2:
                input_values['Amount'] = st.number_input('Amount (valor da transação)', value=0.0, format="%.2f")

            # Criar inputs para as features V1 a V28 de forma dinâmica
            st.markdown("---") # Divisor visual
            st.write("**Features PCA (V1-V28)**")
            v_cols = st.columns(4) # Usar 4 colunas para as features V
            for i in range(1, 29):
                col = v_cols[(i - 1) % 4]
                with col:
                    input_values[f'V{i}'] = st.number_input(f'V{i}', value=0.0, format="%.6f", key=f'v{i}')
            st.markdown("---")

            # --- Botão e Lógica de Inferência ---
            if st.button('🔍 Analisar Transação', type="primary"):
                
                # 1. Organiza os dados na ordem correta usando a lista de features
                input_data_list = [input_values[name] for name in feature_names]
                
                # 2. Cria um DataFrame do Pandas com os nomes das colunas
                #    Este é o formato que o CatBoost (e outros modelos) espera.
                input_df = pd.DataFrame([input_data_list], columns=feature_names)
                
                st.write("Dados de entrada para o modelo:")
                st.dataframe(input_df)

                # 3. Faz a predição usando o DataFrame
                prediction = model.predict(input_df.values)[0]
                
                # 4. Obtém as probabilidades
                try:
                    input_df = input_df.values
                    probabilities = model.predict_proba(input_df)[0]
                    prob_fraude = probabilities[1] # Probabilidade da classe "1" (Fraude)
                    prob_nao_fraude = probabilities[0] # Probabilidade da classe "0" (Não Fraude)
                except Exception as e:
                    st.warning(f"Não foi possível obter as probabilidades: {e}")
                    prob_fraude, prob_nao_fraude = 0, 0


                # 5. Exibe o resultado de forma clara
                st.subheader("📊 Resultado da Análise")
                if prediction == 1:
                    st.error(f"**Predição: Fraude Detectada** (Classe: {prediction})")
                else:
                    st.success(f"**Predição: Transação Normal** (Classe: {prediction})")
                

        except FileNotFoundError:
            st.sidebar.error(f"ERRO: O arquivo do modelo não foi encontrado. Verifique o caminho.")
            st.stop()
        except Exception as e:
            st.sidebar.error(f"ERRO ao carregar ou usar o modelo: {e}")
            st.stop()
    else:
        st.info("👆 Por favor, selecione um modelo no menu acima para iniciar.")

def chart_page():
    """
    Cria uma página no Streamlit para análise exploratória,
    fazendo o download do dataset sob demanda.
    """
    st.title("Página de Análise com Gráficos Nativos")
    st.markdown("Esta página exibe visualizações para análise de dados usando exclusivamente `st.bar_chart`, `st.area_chart` e `st.dataframe`.")

    DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

    # --- Carregamento dos Dados ---
    @st.cache_data
    def load_data(url):
        try:
            with st.spinner("Baixando e carregando o dataset (aprox. 150MB)... Isso pode levar alguns segundos."):
                dados = pd.read_csv(url)
                return dados
        except Exception as e:
            st.error(f"Ocorreu um erro ao baixar ou carregar os dados: {e}")
            return None

    # Chama a função para carregar os dados
    dados = load_data(DATA_URL)

    if dados is None:
        st.error("Não foi possível carregar os dados. Verifique sua conexão com a internet e tente novamente.")
        return

    if st.checkbox("Mostrar tabela de dados (head)"):
        st.subheader("Visualizando as primeiras linhas")
        st.dataframe(dados.head())

    if st.checkbox("Mostrar estatísticas descritivas"):
        st.subheader("Estatísticas Descritivas")
        st.write(dados.describe())

    st.markdown("---")
    st.subheader("Análise da Variável Alvo (`Class`)")

    st.write("Contagem de Transações Normais (0) vs. Fraudulentas (1)")
    class_counts = dados['Class'].value_counts()
    st.bar_chart(class_counts)
    st.info(f"""
    O dataset é extremamente desbalanceado:
    - **Transações Normais (0):** {class_counts[0]}
    - **Transações Fraudulentas (1):** {class_counts[1]}
    - Apenas **{class_counts[1]/len(dados)*100:.4f}%** das transações são fraudes.
    """)

    st.markdown("---")
    st.subheader("Análise das Features `Time` e `Amount` (Histogramas)")
    st.write("Os gráficos de área abaixo funcionam como histogramas, mostrando a frequência de transações agrupadas por `Time` e `Amount`.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Distribuição da Feature `Time`")
        hist_time_values, hist_time_bins = np.histogram(dados['Time'], bins=50)
        time_chart_data = pd.DataFrame({'Frequência': hist_time_values}, index=hist_time_bins[:-1])
        st.area_chart(time_chart_data)

    with col2:
        st.markdown("#### Distribuição da Feature `Amount`")
        filtered_amount = dados[dados['Amount'] < 1000]['Amount']
        hist_amount_values, hist_amount_bins = np.histogram(filtered_amount, bins=50)
        amount_chart_data = pd.DataFrame({'Frequência': hist_amount_values}, index=hist_amount_bins[:-1])
        st.area_chart(amount_chart_data)

def main():
    add_selectbox = st.sidebar.selectbox(
        "Features",
        ("Modelos", "Graficos")
    )
    
    if add_selectbox == "Modelos":
        model_page()
    elif add_selectbox == "Graficos":
        chart_page()

    
main()