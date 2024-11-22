import os
import json
import pandas as pd
import traceback
from langchain_community.chat_models import ChatOpenAI
from src.mcqgen.utils import read_file,get_table_data
from src.mcqgen.logger import logging
import streamlit as st


with open("C:\Users\dhruv\mcqgenerator\Response.json","r") as file:
    RESPONSE_JSON = json.load(file)

st.title("MCQ Generator app with langchain")

with st.form("user.inputs"):
    upload_file=st.file_uploader("Upload a PDF or TEXT file")

    mcq_count=st.number_input("No. of MCQs",min_value=3, max_value=50)

    subject=st.text_input("Input Subject", max_chars=20)

    tone=st.text_input("Enter difficulty level", max_chars=20, placeholder="simple")

    button=st.form_submit_button("Generate MCQs")

    if button and upload_file is not None and mcq_count and subject and tone:
        with st.spinner("loading....."):
            try:
                text=read_file(upload_file)

                with get_openai_callback() as cb:
                    response=generate_evaluate_chain(
                        {
                            "text": text,
                            "subject": subject,
                            "number": mcq_count,
                            "tone": tone,
                            "response_json": json.dumps(RESPONSE_JSON)
                        }
                    )

            except Exception as e:
                traceback.print_exception(type(e),e,e.__traceback__)
                st.error("Error")

            else:
                print(f"Total Tokens:{cb.total_tokens}")
                print(f"Prompt Tokens:{cb.prompt_tokens}")
                print(f"Completion Tokens:{cb.completion_tokens}")
                print(f"Total Cost:{cb.total_cost}")
                if isinstance(response,dict):
                    quiz=response.get("quiz", None)
                    if quiz is not None:
                        table_data=get_table_data(quiz)
                        if table_data is not None:
                            df=pd.DataFrame(table_data)
                            df.index+=1
                            st.table(df)
                            st.text_area(label="Review", value=response["review"])
                        else:
                            st.error("Error in the table")

                    
                else:
                    st.write(response)