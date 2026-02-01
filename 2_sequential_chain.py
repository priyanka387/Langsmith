from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import os

load_dotenv()

os.environ['LANGCHAIN_PROJECT'] = 'Sequential LLM App'



prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

config = {
    'tags' : {'llm_app', 'report_generation', 'summarization'},
    'metagdata': {'model': 'gemini-2.5-flash', 'parser': 'StrOutputParser'}
}

result = chain.invoke({'topic': 'Unemployment in India'}, config=config)

print(result)
