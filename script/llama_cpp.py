from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
import os
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.vectorstores.faiss import FAISS
from langchain.chains import RetrievalQA
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM , AutoModelForCausalLM
from langchain.schema.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.llms.llamacpp import LlamaCpp
from langchain.chains.question_answering import load_qa_chain
import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

model_name="../models/chinese-llama-2-7b/ggml-model-f16.gguf"
#model_name="../models/Taiwan-LLM-7B-v2.0-chat/"
embeddings_model_name="../models/text2vec-base-chinese"

now_path=os.path.dirname(os.path.realpath(__file__))
#loader = DirectoryLoader(path=rf"{now_path}/data", glob='**/*.txt')
loader = UnstructuredFileLoader(rf"{now_path}/data/all.txt",encoding="utf-8",autodetect_encoding=True, show_progress=True)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=20, chunk_overlap=20)
split_docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name,model_kwargs={'device':'cuda'})

if os.path.exists(rf"{now_path}/data/my_faiss_store.faiss") == False:
    vector_store = FAISS.from_documents(split_docs, embeddings)
else:
    vector_store = FAISS.load_local(rf"{now_path}/data/my_faiss_store.faiss",embeddings=embeddings)
    
#tokenizer = AutoTokenizer.from_pretrained(model_name)

#base_model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.float16,device_map='auto',trust_remote_code=True)

#llm = base_model.eval()

    
_chat_llm = LlamaCpp(
            model_path=model_name,
            max_tokens=500,
            n_gpu_layers= 40,
            n_batch= 256 ,
            verbose=False,
            n_ctx=1024,
            callbacks=[StreamingStdOutCallbackHandler()],
        )


chain=load_qa_chain(llm=_chat_llm,chain_type="stuff")

while True:
    query = input("問題: ")
    docs=vector_store.search(query=query, search_kwargs={"score_threshold": .5,"k": 3})
    print(docs)
    print("\n")
    chain.run(input_documents=docs,query=query)
