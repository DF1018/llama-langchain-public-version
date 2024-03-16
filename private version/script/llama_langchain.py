from torch import cuda, bfloat16
import transformers
import time
import yaml
from langchain.chains import RetrievalQA
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.vectorstores.faiss import FAISS
from pathlib import Path
from langchain.document_loaders import TextLoader
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import load_tools
from langchain.llms import HuggingFacePipeline
from langchain.agents import AgentOutputParser
from langchain.prompts import PromptTemplate
from langchain.agents.conversational_chat.prompt import FORMAT_INSTRUCTIONS
from langchain.output_parsers.json import parse_json_markdown
from langchain.schema import AgentAction, AgentFinish
from langchain.agents import initialize_agent
from langchain.agents import AgentType, Tool, initialize_agent
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, pipeline
from langchain.memory import ConversationBufferMemory
from langchain.schema import prompt
import torch
import textwrap
import pprint
import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import time

model_path = '../models/Taiwan-LLM-7B-v2.1-chat'
content_path = './data/all.txt'
embeddings_model_path="../models/text2vec-base-chinese/"


device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)


model_config = transformers.AutoConfig.from_pretrained(model_path)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto'
)
model.eval()
print(f"Model loaded on {device}")


tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

streamer = TextStreamer(tokenizer, skip_prompt=True)

generate_text = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    temperature=0.01,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=512,  # mex number of tokens to generate in the output
    repetition_penalty=1.1,  # without this output begins repeating
    device_map='auto',
    streamer=streamer
)


llm = HuggingFacePipeline(pipeline=generate_text)


loader = TextLoader(content_path)
documents = loader.load()
text_splitter = CharacterTextSplitter(separator='\n',chunk_size=100, chunk_overlap=20)
texts = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_path,model_kwargs={'device':'cuda'})

docsearch = Chroma.from_documents(texts, embeddings)

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

system_prompt = '''
假裝你是鴻海科技公司董事長劉揚偉，你會根據以下文章去回答問題，不用說多餘的話，只根據以下文章去回答問題。請用繁體中文回答問題。
''' 

def get_prompt(instruction, new_system_prompt ):
    _new_system_prompt = B_SYS + new_system_prompt + E_SYS
    prompt_template =  B_INST + _new_system_prompt + instruction + E_INST
    return prompt_template

instruction = """
{context} \n
Question: {question}
"""

prompt_template= get_prompt(instruction, system_prompt)

_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
        )


chain_type_kwargs = {'prompt': _prompt}

retriever = docsearch.as_retriever(search_kwargs={"k": 1})

ruff = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs,
    verbose=True
)



while True:
    input_text = input("User: ")
    start = time.process_time()
   
    response=ruff({"query":input_text})
    print("\n")
    print(response)
    
    end = time.process_time()
    print("執行時間：%f 秒" % (end - start))
