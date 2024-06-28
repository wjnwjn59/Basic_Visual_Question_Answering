import os
import base64
import streamlit as st
from annotated_text import annotated_text

from components.streamlit_footer import footer
from config.model_config import QA_Config
from models.qa_model import get_model
from database import getEmbedding
import sqlite3
import numpy as np
import random

EXAMPLE_QUESTIONS = [
    "What doctor was with Chopin when he wrote out his will?",
    "Where was Chopin invited to in late summer?",
    "What city did Chopin perform at on September 27?",
    "What did Chopin write while staying with Doctor Adam Łyszczyński?",
    "When did Chopin last appear in public?",
    "Who were the beneficiaries of his last public concert?",
    "What was the diagnosis of Chopin's health condition at this time?",
    "Where was Chopin's last public performance?",
    "Who did Chopin play for while she sang?",
    "In 1849 where did Chopin live?",
    "Who was anonymously paying for Chopin's apartment?",
    "When did Chopin return to Paris?",
    "Chopin accompanied which singer for friends?",
    "Where did his friends found Chopin an apartment in 1849?",
    "Who paid for Chopin's apartment in Chaillot?",
    "When did Jenny Lind visit Chopin?",
    "When did his sister come to stay with Chopin?",
    "Who did Kerry publicize as a lesbian while discussing gay rights, some time after the debate?",
    "Why was there tension the day after the election?",
    "Who was the lone supporter of the motion, from the Senate?",
    "What did Kerry say affected the ability to know if the results of the Ohio vote was unbiased?",
    "What was disallowed in advertising during the two months prior to the general election?",
    "What action suggested by a state, would have affecting the outcome of the electoral votes?",
    "Who was the Liberal Party of Australia's longest-serving leader?",
    "How does grain become malted?",
    "What company acquired the Anheuser-Busch brewing Company in 2008?",
    "What is a professional called at a restaurant who advises customers about beer and food pairs?",
    "What do brewing companies sometimes use to give more alcohol content to their beer?",
    
]

EXAMPLE_QUESTION = ""
def replace_input_text():
    EXAMPLE_QUESTION = random.choice(EXAMPLE_QUESTIONS)
    st.session_state.question = EXAMPLE_QUESTION



def get_answer(question, context):
    qa_model = get_model(QA_Config.model_id)
    result_dict = qa_model(question=question, context=context)
    
    answer_text = result_dict['answer'] if 'answer' in result_dict else "No answer found"
    answer_start = result_dict['start'] if 'start' in result_dict else -1
    answer_end = result_dict['end'] if 'end' in result_dict else -1
    
    start_ans_idx = answer_start
    end_ans_idx = answer_end
    
    return [
        context[:start_ans_idx],
        (context[start_ans_idx:end_ans_idx], '', '#afa'),
        context[end_ans_idx:]
    ]
 
def get_nearest_contexts(query_embedding, db_file='embeddings.db'):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Fetch data from the database
    cursor.execute('''SELECT context, question_embedding FROM embeddings''')
    all_embeddings = cursor.fetchall()

    # Calculate similarity scores (Euclidean distance)
    similarities = [(context, np.linalg.norm(np.frombuffer(embedding) - query_embedding))
                    for context, embedding in all_embeddings]

    # Sort by similarity and get the nearest context
    nearest_context = min(similarities, key=lambda x: x[1])[0]  # Get the context with the minimum distance

    # Close connection
    conn.close()

    return nearest_context    


    
def main():
    st.set_page_config(page_title="Question Answering Demo - AI VIETNAM",
                       page_icon='static/aivn_favicon.png',
                       layout="wide")

    db_file = 'embeddings.db'
    if not os.path.exists(db_file):
        # If the database file doesn't exist, create it and populate with embeddings dataset
        embeddings_dataset = getEmbedding.BuildVectorDB()
        getEmbedding.CreateDB(embeddings_dataset, db_file)
    else:
        print("Database file already exists.")
        
        
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
        # input_context = st.text_area('__Context__',
        #                              height=100,
        #                              max_chars=1000,
        #                              key='context',
        #                              placeholder='Input some text...')
        
        col1, col2 = st.columns([1, 2])
        with col1:
            submission = st.form_submit_button('Submit')
        with col2:
            example_button = st.form_submit_button('Run example', 
                                                   on_click=replace_input_text)
            
        if example_button:
            st.divider()
            if input_question == '':
                st.write('__Error:__ Input question cannot be empty!')
            else:
                input_quest_embedding = getEmbedding.get_embeddings([input_question]).cpu().detach().numpy()[0]

                nearest_contexts = get_nearest_contexts(input_quest_embedding)
                
                if nearest_contexts:
                    nearest_context = nearest_contexts  # Assuming the second element is the nearest context
                    st.text_area("The context for your question: ", nearest_context)
                    result_lst = get_answer(input_question, nearest_context)
                    st.write(f'__Answer__: {result_lst[1][0]}')
                    annotated_text(result_lst)
                else:
                    st.write('No similar context found in the database.')
                    

        if submission:
            st.divider()
            if input_question == '':
                st.write('__Error:__ Input question cannot be empty!')
            else:
                input_quest_embedding = getEmbedding.get_embeddings([input_question]).cpu().detach().numpy()[0]

                nearest_contexts = get_nearest_contexts(input_quest_embedding)
                
                if nearest_contexts:
                    nearest_context = nearest_contexts  # Assuming the second element is the nearest context
                    st.text_area("The context for your question: ", nearest_context)
                    result_lst = get_answer(input_question, nearest_context)
                    st.write(f'__Answer__: {result_lst[1][0]}')
                    annotated_text(result_lst)
                else:
                    st.write('No similar context found in the database.')

    footer()


if __name__ == '__main__':
    main()