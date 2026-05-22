# -*- coding: utf-8 -*-
"""
Created on Thu May 21 15:17:56 2026

@author: Vicenzo
"""


import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson, norm

st.set_page_config(page_title="Dimensionamento de Sobressalentes", layout="wide")


def calcular_poisson(lmbda, n, t, risco_alvo):
    m = lmbda * n * t
    x = 0
    prob_acumulada = 0
    x_ideal = -1
    
    lista_x, lista_p, lista_margem, lista_risco = [], [], [], []
    
    while True:
        p_x = poisson.pmf(x, m)
        prob_acumulada += p_x
        risco_atual = max(1 - prob_acumulada, 0.0)
        
        lista_x.append(x)
        lista_p.append(p_x)
        lista_margem.append(prob_acumulada)
        lista_risco.append(risco_atual)
        
        if risco_atual < risco_alvo and x_ideal == -1:
            x_ideal = x
            
        if x_ideal != -1 and x >= x_ideal + 1:
            break
        x += 1
        
    df = pd.DataFrame({
        'x': lista_x,
        'P(X=x)': lista_p,
        'Margem Seg.': lista_margem,
        'Risco': lista_risco })
    
    return df, x_ideal, m

def calcular_normal(lmbda, n, t, risco_alvo):
    m = lmbda * n * t
    sigma = np.sqrt(m)
    x = 0
    x_ideal = -1
    
    lista_x, lista_p, lista_margem, lista_risco = [], [], [], []
    
    while True:
        prob_acum = norm.cdf(x, loc=m, scale=sigma)
       
        if x == 0:
            p_x = prob_acum
        else:
            p_x = prob_acum - norm.cdf(x - 1, loc=m, scale=sigma)
            
        risco_atual = max(1 - prob_acum, 0.0)
        
        lista_x.append(x)
        lista_p.append(p_x)
        lista_margem.append(prob_acum)
        lista_risco.append(risco_atual)
        
        if risco_atual < risco_alvo and x_ideal == -1:
            x_ideal = x
            
        if x_ideal != -1 and x >= x_ideal + 1:
            break
        x += 1
        
    df = pd.DataFrame({
        'x': lista_x,
        'P(X=x)': lista_p,
        'Margem Seg.': lista_margem,
        'Risco': lista_risco })
    
    return df, x_ideal, sigma

st.title(" Sistema de Dimensionamento de Sobressalentes")
st.write("Insira os parâmetros na barra lateral esquerda para calcular as recomendações.")


st.sidebar.header("Parâmetros de Entrada")

L = st.sidebar.number_input("Lambda (taxa de falha):", min_value=0.0000, value=0.05, step=0.01, format="%.6f")
N = st.sidebar.number_input("Número de máquinas ativas (n):", min_value=1, value=10, step=1)
T = st.sidebar.number_input("Tempo (t):", min_value=0.1, value=1.0, step=0.1)
R_PCT = st.sidebar.number_input("Risco máximo aceitável em %:", min_value=0.01, max_value=100.0, value=5.0, step=1.0)

calcular = st.sidebar.button("Calcular Dimensionamento")

if calcular:
    R_ALVO = R_PCT / 100.0
    n_10PCT = round(N * 0.1)
    
    # Cálculo do lambda geral conforme solicitado
    lambda_geral = L * N

    df_p, x_p, m_val = calcular_poisson(L, N, T, R_ALVO)

    st.divider() 
    col1, col2, col3 = st.columns(3)
    col1.metric("Média (m)", f"{m_val:.2f}")
    col2.metric("Risco Alvo", f"{R_PCT}%")
    col3.metric("Regra dos 10%", f"{n_10PCT} peças")

    st.divider()

    
    def exibir_resumo_streamlit(df, x_alvo, titulo):
        st.subheader(titulo)
        
        idx_inicio = max(0, x_alvo - 1)
        resumo = df.iloc[idx_inicio : x_alvo + 2].copy()
        
        resumo['P(X=x)'] = resumo['P(X=x)'].apply(lambda v: f"{v:.4%}")
        resumo['Margem Seg.'] = resumo['Margem Seg.'].apply(lambda v: f"{v:.4%}")
        resumo['Risco'] = resumo['Risco'].apply(lambda v: f"{v:.4%}")
        
        st.success(f"**Quantidade Recomendada:** {x_alvo} peças")
        st.dataframe(resumo, use_container_width=True, hide_index=True)

    col_tabela1, col_tabela2 = st.columns(2)
    
    with col_tabela1:
        exibir_resumo_streamlit(df_p, x_p, "Distribuição de Poisson")
        
    with col_tabela2:
        # Condicional para exibir a Aproximação pela Normal apenas se o critério for atendido
        if lambda_geral >= 20:
            df_n, x_n, sigma_val = calcular_normal(L, N, T, R_ALVO)
            exibir_resumo_streamlit(df_n, x_n, "Aproximação pela Normal")
        else:
            st.subheader("Aproximação pela Normal")
            st.warning(f"**Cálculo não recomendado:** O λ geral é **{lambda_geral:.2f}**. O método de Aproximação pela Normal é válido apenas para λ ≥ 20.")

else:
    st.info("👈 Preencha os valores na barra lateral e clique em 'Calcular Dimensionamento'!")