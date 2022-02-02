import pickle
import streamlit as st
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import LabelEncoder
label_enc = LabelEncoder()



# clean_doc=pickle.load(open('clean_doc.obj','rb'))
tfid= pickle.load(open('Tfidfmodels.pkl','rb'))
model=pickle.load(open('kmeanmodel.pkl','rb'))
data= pd.read_csv('https://raw.githubusercontent.com/EngrAhmadUmar/DataMining/main/news.csv')
      
# front end elements of the web page 
html_temp = """ 
<div style ="background-color:yellow;padding:13px"> 
<h1 style ="color:black;text-align:center;">Streamlit New Article Clustering App</h1> 
</div> 
""" 
st.markdown(html_temp, unsafe_allow_html = True) 
default_value_goes_here = ""
Content = st.text_area("label goes here", default_value_goes_here)
result =""
data= pd.read_csv('https://raw.githubusercontent.com/Diane10/news_classifier/main/All_combined_New_papers%20-%20Sheet1.csv')
data["label"] = label_enc.fit_transform(data[["label"]])  
# when 'Predict' is clicked, make the prediction and store it 
if st.button("Predict"): 
  pred = model.predict(tfid.transform([Content]))
  if pred==1:
    st.write('This article is about culture, celebreties or art.')   
    pred= int(pred)
    data_pred = data.loc[(data['label'] == pred)]
    st.dataframe(data_pred['source_url'].unique())
  elif pred==0:
    st.write('Article is a business article')
    term="business"
    pred= int(pred)
    data_pred = data.loc[(data['label'] == pred)]
    # result_df= data_pred[data_pred['source_url'].str.contains(term)]
    st.dataframe(data_pred['source_url'].unique())
  elif pred==2:
    st.write('Article is a political article') 
    pred= int(pred)
    term="pol"
    pred= int(pred)
    data_pred = data.loc[(data['label'] == pred)]
    # result_df= data_pred[data_pred['source_url'].str.contains(term)]
    st.dataframe(data_pred['source_url'].unique())
  elif pred==3:
    st.write('Article is a sport article')
    pred= int(pred)
    term='sport'
    data_pred = data.loc[(data['label'] == pred)]
    # result_df= data_pred[data_pred['source_url'].str.contains(term)]
    st.dataframe(data_pred['source_url'].unique())
