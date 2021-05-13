import streamlit as st
import pickle
from nltk.stem.snowball import SnowballStemmer
stemmer=SnowballStemmer('english')
vectzr=pickle.load(open('Text_Vectorizer.pkl','rb'))
clf=pickle.load(open('Text_classfier.pkl','rb'))
st.title('Comment Sentiment Predictor')
st.write('This is a web app based on a Machine learning algorithm, which PREDICTS IF A PERSON HAS LIKE OR DISLIKE TOWARDS A MOVIE, given their comment on the respective movie.')
st.write('Write a comment on a movie that you wish, you want to comment in the below space.The comment shall not be too short and it shall only be in English.')
#st.write('A model comment by person on a movie, has already been typed in for your ease of finding.You can erase it and type your own comment.')
comment=st.text_area('Your comment goes here (A model comment has been typed.You can erase it and type your comment.)',"The movie largely delivers, splashing its ambitious three-hour narrative across a sprawling canvas of characters, eras, and not-quite-insurmountable challenges.")
#pre-processing
punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
comment=''.join([x for x in comment if x not in punctuations if not x.isdigit()])
comment=' '.join([stemmer.stem(word) for word in (comment).split()])
#vectoriztion
if st.button('Predict'):
    if clf.predict(vectzr.transform([comment]).toarray())==[1]:
        st.write('The person who wrote the comment, probably LIKED the respective movie.')
    else:
        st.write("The person who wrote the comment, probably DIDN'T LIKE the respective movie.")
st.markdown("<br><br><br>",unsafe_allow_html=True)
st.subheader("SIGNIFICANCE OF THE ALGORITHM")
st.markdown("When there is huge data of text contents, it becomes a tedious and time consuming work for humans to classify each and every text.This is where the algorithm comes in handy, when it could read and classify thousands of texts within a phenomenal amount of time with an apreciable accuracy.")
st.markdown("**Have suggestions or comments for me ?**")
st.write("**Found a wrong prediction?**")
st.write("**You can mail me at kotiravin@gmail.com or call me at +91 9384759214**")
