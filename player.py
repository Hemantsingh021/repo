import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
#st.title('Player Performance Prediction App')
st.markdown(
    """
    <style>
    .stApp {
        background-color: brown;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    "<h1 style='text-align: center; color:white; font-size: 3em;'> Player Performance Prediction App </h1>",
    unsafe_allow_html=True
)
df= pd.read_csv('player.csv')
df.drop('Player', axis= 'columns',inplace=True)
df.drop('Team', axis= 'columns',inplace=True)
df.drop('Role', axis= 'columns',inplace=True)
df.drop('Runs', axis= 'columns',inplace=True)
df.drop('Wickets', axis= 'columns',inplace=True)
df['Fitness']=df['Fitness'].apply(lambda x: 1 if x=='Fit'  else 0)
x= df.drop('FantasyScore',axis= 'columns')
y= df['FantasyScore']

model= LinearRegression()
model.fit(x,y)
lr=st.number_input('Enter average number of runs in last 5 matches')
lw= st.number_input('Enter average number of wickets in last 5 matches')
ar=st.number_input('Enter number of runs in last match')
aw=st.number_input('Enter number of wickets in last match')
f=st.number_input('Enter fitness score (if fit then 1 else 0)')
rf=st.number_input('Enter recent form score')
#r=st.number_input('Enter number of runs in the match played')
#w=st.number_input('Enter number of wickets in the match played')
input_data= [[lr,lw,ar,aw,f,rf,]]
prediction= model.predict(input_data)
# m= model.coef_
# c=model.intercept_
# m1= m[0]
# m2= m[1]
# m3= m[2]
# m4= m[3]
# m5= m[4]
# m6= m[5]
# m7= m[6]
# m8= m[7]
# prediction= m1*lr+m2*lw+m3*ar+m4*aw+m5*f+m6*rf+m7*r+m8*w +c
if st.button('Predict'):
    st.write(f'The predicted fantasy score is {prediction[0]:.2f}')
    st.balloons()
    st.markdown(
        "<h5 style='text-align: center; color: #424242;'>Thank you for using the app 🎉🎊</h5>",
        unsafe_allow_html=True
    )
