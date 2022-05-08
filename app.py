from io import BytesIO

import streamlit as st
from PIL import Image

import proj_utils as utils

repo = utils.open_meme_repo('meme_repo.pkl')

st.header('✨ spicy meme machine 5000™ ✨')
st.caption('oh, hello there. let\'s make you a spicy meme. the spicy meme machine 5000™ uses top memes scraped from '
           'reddit to attempt to memeify any image. (hey, key word here is attempt. don\'t come here expecting comedy '
           'gold or nothin\'.)')

upload_method = st.radio('how would you like to provide your base image?', ('url', 'file upload'))


def memeify(url=None, raw=None, sample=None):
    try:
        with st.empty():
            st.text('meme magic happening... 🎩🪄✨')
            memes, matches = utils.generate_memes(repo, img_url=url, img_raw=raw, sample=sample, get_top_match_urls=True)
            st.subheader('wow, look at these fresh memes we cooked up for u friend')

        for meme in memes:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.image(meme)
            buffer = BytesIO()
            meme.save(buffer, format='JPEG')
            byte_img = buffer.getvalue()
            with col2:
                st.download_button(
                    label='download this ~spicy meme~',
                    data=byte_img,
                    mime='image/jpeg',
                )

        st.subheader('top memes used in the model for this base image:')
        cols = st.columns(len(matches))
        for i, col in enumerate(cols):
            with col:
                st.image(matches.iloc[i])

    except Exception as e:
        st.text('well this embarrassing, that didn\'t work.\nmaybe try a bigger sample or a different image?')
        raise e


if upload_method == 'url':
    url = st.text_input('image url', 'https://')
    s = st.slider('how many memes should we use in the corpus of the markov chain generator?', 25, 300, 100)
    if st.button('meme me'):
        memeify(url=url, sample=s)

elif upload_method == 'file upload':
    uploaded_file = st.file_uploader('choose a file')
    s = st.slider('how many memes should we use in the corpus of the markov chain generator?', 25, 300, 100)
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        if st.button('meme me'):
            memeify(raw=bytes_data, sample=s)
