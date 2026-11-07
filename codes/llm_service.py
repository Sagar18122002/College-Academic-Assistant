
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def get_answer_chain():

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is missing from your .env file."
        )

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=api_key,
        temperature=0
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a College Academic Assistant.

            Answer using only the supplied context.
            Do not invent policies, dates, fees, or facts.
            If the context does not contain the answer, say that
            the available documents do not specify it.

            The supplied policy document contains sample information
            and is not verified as an official college policy.

            Context:
            {context}"""
        ),
        ("human", "{question}")
    ])

    return prompt | llm | StrOutputParser()
