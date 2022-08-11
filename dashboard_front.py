import streamlit as st
import pandas as pd
import numpy as np
import urllib.parse
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.cluster import KMeans
import shap
import plotly.express as px
from zipfile import ZipFile
from sklearn.cluster import KMeans
import requests
import joblib
from xgboost import XGBClassifier


plt.style.use('fivethirtyeight')


# sns.set_style('darkgrid')
def clean(data) :
    data = str(data)
    data = data.replace("[[", "(")
    data = data.replace("]]", ")")
    data = data.replace(" ", "")
    return data


@st.cache
def load_data():
    z = ZipFile("data/default_risk.zip")
    data = pd.read_csv(z.open('default_risk.csv'), index_col='SK_ID_CURR', encoding='utf-8')

    z = ZipFile("data/X_sample.zip")
    sample = pd.read_csv(z.open('X_sample.csv'), index_col='SK_ID_CURR',encoding='utf-8')

    description = pd.read_csv("data/features_description.csv",
                              usecols=['Row', 'Description'], index_col=0, encoding='unicode_escape')

    target = data.iloc[:, -1:]

    return data, sample, target, description


def api_request(API_url, id):
    response = requests.get(API_url + "?data=" + str(id))
    return response.json()


@st.cache(allow_output_mutation=True)
def load_knn(sample):
    knn = knn_training(sample)
    return knn


@st.cache
def load_infos_gen(data):
    lst_infos = [data.shape[0],
                 round(data["AMT_INCOME_TOTAL"].mean(), 2),
                 round(data["AMT_CREDIT"].mean(), 2)]

    nb_credits = lst_infos[0]
    rev_moy = lst_infos[1]
    credits_moy = lst_infos[2]

    targets = data.TARGET.value_counts()

    return nb_credits, rev_moy, credits_moy, targets


def identite_client(data, id):
    data_client = data[data.index == int(id)]
    return data_client


@st.cache
def load_age_population(data):
    data_age = round((data["DAYS_BIRTH"] / 365), 2)
    return data_age


@st.cache
def load_income_population(sample):
    df_income = pd.DataFrame(sample["AMT_INCOME_TOTAL"])
    df_income = df_income.loc[df_income['AMT_INCOME_TOTAL'] < 200000, :]
    return df_income


@st.cache
def load_prediction(API_URL, id):
    prediction = api_request(API_URL, id)
    print("PREDICTIONNNNNNNN :::::", prediction)
    return prediction


@st.cache
def load_kmeans(sample, id, mdl,data):
    index = sample[sample.index == int(id)].index.values
    index = index[0]
    data_client = pd.DataFrame(sample.loc[sample.index, :])
    df_neighbors = pd.DataFrame(mdl.fit_predict(data_client), index=data_client.index)
    df_neighbors = pd.concat([df_neighbors, data], axis=1)
    return df_neighbors.iloc[:, 1:].sample(10)


@st.cache
def knn_training(sample):
    knn = KMeans(n_clusters=2).fit(sample)
    return knn


def main():
    # Loading data……
    data, sample, target, description = load_data()
    id_clients = sample.index.values
    model = joblib.load(open('model/XGBClassifier.pkl', 'rb')).steps[1][1]
    import streamlit.components.v1 as components
    #######################################
    # SIDEBAR
    #######################################
    API_url = "https://openclassrooms-api.herokuapp.com/predict"
    local_url = "http://127.0.0.1:5000/predict"
    # Title display
    html_temp = """
    <div style="background-color: tomato; padding:10px; border-radius:10px">
    <h1 style="color: white; text-align:center">Dashboard Scoring Credit</h1>
    </div>
    <p style="font-size: 20px; font-weight: bold; text-align:center">Credit decision support…</p>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Customer ID selection
    st.sidebar.header("**General Info**")

    # Loading selectbox
    chk_id = st.sidebar.selectbox("Client ID", id_clients)
    data_to_predict = sample[sample.index == int(chk_id)].drop(['TARGET'], axis=1).to_numpy()
    data_str = str(data_to_predict)
    data_http = urllib.parse.quote_plus(data_str)
    response = requests.get(API_url+"?data="+data_http)
    response = response.json()
    prediction = response["prediction"]
    if prediction == 0.0:
        st.write("Loan GRANTED !")
    else:
        st.write("Loan DENIED !")
    # Loading general info
    nb_credits, rev_moy, credits_moy, targets = load_infos_gen(data)

    ### Display of information in the sidebar ###
    # Number of loans in the sample
    st.sidebar.markdown("<u>Number of loans in the sample :</u>", unsafe_allow_html=True)
    st.sidebar.text(nb_credits)

    # Average income
    st.sidebar.markdown("<u>Average income (USD) :</u>", unsafe_allow_html=True)
    st.sidebar.text(rev_moy)

    # AMT CREDIT
    st.sidebar.markdown("<u>Average loan amount (USD) :</u>", unsafe_allow_html=True)
    st.sidebar.text(credits_moy)

    # PieChart
    # st.sidebar.markdown("<u>......</u>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    plt.pie(targets, explode=[0, 0.1], labels=['No default', 'Default'], autopct='%1.1f%%', startangle=90)
    st.sidebar.pyplot(fig)

    #######################################
    # HOME PAGE - MAIN CONTENT
    #######################################
    # Display Customer ID from Sidebar
    st.write("Customer ID selection :", chk_id)

    # Customer information display : Customer Gender, Age, Family status, Children, …
    st.header("**Customer information display**")

    if st.checkbox("Show customer information ?"):

        infos_client = identite_client(data, chk_id)
        st.write("**Gender : **", infos_client["CODE_GENDER"].values[0])
        st.write("**Age : **{:.0f} ans".format(int(infos_client["DAYS_BIRTH"] / 365)))
        st.write("**Family status : **", infos_client["NAME_FAMILY_STATUS"].values[0])
        st.write("**Number of children : **{:.0f}".format(infos_client["CNT_CHILDREN"].values[0]))

        # Age distribution plot
        data_age = load_age_population(data)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data_age, edgecolor='k', color="goldenrod", bins=20)
        ax.axvline(int(infos_client["DAYS_BIRTH"].values / 365), color="green", linestyle='--')
        ax.set(title='Customer age', xlabel='Age(Year)', ylabel='')
        st.pyplot(fig)

        st.subheader("*Income (USD)*")
        st.write("**Income total : **{:.0f}".format(infos_client["AMT_INCOME_TOTAL"].values[0]))
        st.write("**Credit amount : **{:.0f}".format(infos_client["AMT_CREDIT"].values[0]))
        st.write("**Credit annuities : **{:.0f}".format(infos_client["AMT_ANNUITY"].values[0]))
        st.write("**Amount of property for credit : **{:.0f}".format(infos_client["AMT_GOODS_PRICE"].values[0]))

        # Income distribution plot
        data_income = load_income_population(data)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data_income["AMT_INCOME_TOTAL"], edgecolor='k', color="goldenrod", bins=10)
        ax.axvline(int(infos_client["AMT_INCOME_TOTAL"].values[0]), color="green", linestyle='--')
        ax.set(title='Customer income', xlabel='Income (USD)', ylabel='')
        st.pyplot(fig)

        # Relationship Age / Income Total interactive plot
        data_sk = data.reset_index(drop=False)
        data_sk.DAYS_BIRTH = (data_sk['DAYS_BIRTH'] / 365).round(1)
        fig, ax = plt.subplots(figsize=(10, 10))
        fig = px.scatter(data_sk, x='DAYS_BIRTH', y="AMT_INCOME_TOTAL",
                         size="AMT_INCOME_TOTAL", color='CODE_GENDER',
                         hover_data=['NAME_FAMILY_STATUS', 'CNT_CHILDREN', 'NAME_CONTRACT_TYPE', 'SK_ID_CURR'])

        fig.update_layout({'plot_bgcolor': '#f0f0f0'},
                          title={'text': "Relationship Age / Income Total", 'x': 0.5, 'xanchor': 'center'},
                          title_font=dict(size=20, family='Verdana'), legend=dict(y=1.1, orientation='h'))

        fig.update_traces(marker=dict(line=dict(width=0.5, color='#3a352a')), selector=dict(mode='markers'))
        fig.update_xaxes(showline=True, linewidth=2, linecolor='#f0f0f0', gridcolor='#cbcbcb',
                         title="Age", title_font=dict(size=18, family='Verdana'))
        fig.update_yaxes(showline=True, linewidth=2, linecolor='#f0f0f0', gridcolor='#cbcbcb',
                         title="Income Total", title_font=dict(size=18, family='Verdana'))

        st.plotly_chart(fig)

    else:
        st.markdown("<i>…</i>", unsafe_allow_html=True)

    # Customer solvability display
    st.header("**Customer file analysis**")

    # Compute decision according to the best threshold
    # if prediction <= xx :
    #    decision = "<font color='green'>**LOAN GRANTED**</font>"
    # else:
    #    decision = "<font color='red'>**LOAN REJECTED**</font>"

    # st.write("**Decision** *(with threshold xx%)* **: **", decision, unsafe_allow_html=True)
    st.markdown("<u>Customer Data :</u>", unsafe_allow_html=True)
    st.write(identite_client(data, chk_id))

    # Feature importance / description
    if st.checkbox("Customer ID {:.0f} feature importance ?".format(chk_id)):
        shap.initjs()
        X = sample[sample.index == int(chk_id)].drop(['TARGET'], axis=1)
        number = st.slider("Pick a number of features…", 0, 20, 5)
        #model.fit(sample.drop(['TARGET'], axis=1), sample.TARGET)
        fig, ax = plt.subplots(figsize=(10, 10))
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, X, plot_type="bar", max_display=number, color_bar=False, plot_size=(5, 5))
        st.pyplot(fig)

        if st.checkbox("Need help about feature description ?"):
            list_features = description.index.to_list()
            feature = st.selectbox('Feature checklist…', list_features)
            st.table(description.loc[description.index == feature][:1])

    else:
        st.markdown("<i>…</i>", unsafe_allow_html=True)

    # Similar customer files display
    chk_voisins = st.checkbox("Show similar customer files ?")
    #if btn_predict:
    #    prediction = load_prediction(local_url, id)
    #    if prediction['prediction'] == 0:
    #        st.write('Dossier validé par la banque')
    #    else:
    #        st.write("refused")
    if chk_voisins:
        knn = load_knn(sample)
        st.markdown("<u>List of the 10 files closest to this Customer :</u>", unsafe_allow_html=True)
        st.dataframe(load_kmeans(sample, chk_id, knn, data))
        st.markdown("<i>Target 1 = Customer with default</i>", unsafe_allow_html=True)
    else:
        st.markdown("<i>…</i>", unsafe_allow_html=True)

    st.markdown('***')


if __name__ == '__main__':
    main()
