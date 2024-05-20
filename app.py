import os
import base64
import streamlit as st
from annotated_text import annotated_text

from components.streamlit_footer import footer
from config.model_config import QA_Config
from models.qa_model import get_model

EXAMPLE_QUESTION = 'Where is the highest mountain in solar system located?'
EXAMPLE_CONTEXT = 'The highest mountain and volcano in the Solar System is on the planet Mars. It is called Olympus Mons and is 16 miles (24 kilometers) high which makes it about three times higher than Mt. Everest.'

def get_answer(question, context):
    qa_model = get_model(QA_Config.model_id)
    result_dict = qa_model(question, context)
    start_ans_idx = result_dict['start']
    end_ans_idx = result_dict['end']

    return  [
        context[:start_ans_idx],
        (context[start_ans_idx:end_ans_idx], '', '#afa'),
        context[end_ans_idx:]
    ]

def replace_input_text():
    st.session_state.question = EXAMPLE_QUESTION
    st.session_state.context = EXAMPLE_CONTEXT

def main():
    st.set_page_config(page_title="Question Answering Demo - AI VIETNAM",
                       page_icon='static/aivn_favicon.png',
                       layout="wide")

    col1, col2 = st.columns([0.8, 0.2], gap='large')
    
    with col1:
        st.title(':thought_balloon: :blue[Question Answering] Demo')
        
    with col2:
        logo_img = open("static/aivn_logo.png", "rb").read()
        logo_base64 = base64.b64encode(logo_img).decode()
        st.markdown(
            f"""
            <a href="https://aivietnam.edu.vn/">
                <img src="data:image/png;base64,{logo_base64}" width="full">
            </a>
            """,
            unsafe_allow_html=True,
        )

    with st.form("my_form"):
        input_question = st.text_input('__Question__',
                                        key='question',
                                        max_chars=100,
                                        placeholder='Input some text...')
        input_context = st.text_area('__Context__',
                                     height=100,
                                     max_chars=1000,
                                     key='context',
                                     placeholder='Input some text...')
        
        col1, col2 = st.columns([1, 2])
        with col1:
            submission = st.form_submit_button('Submit')
        with col2:
            example_button = st.form_submit_button('Run example', 
                                                   on_click=replace_input_text)
            
        if example_button:
            st.divider()
            result_lst = get_answer(EXAMPLE_QUESTION, EXAMPLE_CONTEXT)
            st.write(f'__Answer__: {result_lst[1][0]}')
            annotated_text(result_lst)

        if submission:
            st.divider()
            if input_context == '' or input_question == '':
                st.write('__Error:__ Either input question or context cannot be empty!')
            else:
                result_lst = get_answer(input_question, input_context)
                st.write(f'__Answer__: {result_lst[1][0]}')
                annotated_text(result_lst)


    footer()


if __name__ == '__main__':
    main()