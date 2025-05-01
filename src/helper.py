from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import TokenTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer
from src.prompt import *
import os
from dotenv import load_dotenv
import re

load_dotenv()

DEEPSEEK_API_KEY= os.getenv('DEEPSEEK_API_KEY')
HUGGINGFACE_API_KEY=os.getenv('HUGGINGFACE_API_KEY')

os.environ["DEEPSEEK_API_KEY"]=DEEPSEEK_API_KEY
os.environ["HUGGINGFACE_API_KEY"]=HUGGINGFACE_API_KEY

def file_processing(file_path):

    # Load data from PDF
    loader = PyPDFLoader(file_path)
    data = loader.load()

    question_gen = ''

    for page in data:
        question_gen += page.page_content
        
    model_name = "deepseek-ai/deepseek-llm-7b-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Define a token counter using the Hugging Face tokenizer
    def token_counter(text: str) -> int:
        return len(tokenizer.encode(text))

    # Initialize the splitter with the custom token counter
    splitter_ques_gen = TokenTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=10000,
        chunk_overlap=200,
        # length_function=token_counter  # Use the Hugging Face tokenizer here
    )

    # Split the text
    chunks_ques_gen = splitter_ques_gen.split_text(question_gen)
    
    document_ques_gen = [Document(page_content=t) for t in chunks_ques_gen]
    
    #now perform chunkins on document
    splitter_ans_gen = TokenTextSplitter.from_huggingface_tokenizer(
                            tokenizer,
                            chunk_size=1000,
                            chunk_overlap=100)

    # Split the text
    document_ans_gen = splitter_ans_gen.split_documents(document_ques_gen)
    
    return document_ques_gen, document_ans_gen

class LLM_pipeline:
    def __init__(self, file_path):
        self.filepath = file_path
      
    def get_document_quen_ans(self):
        self.document_ques_gen, self.document_answer_gen = file_processing(self.filepath)
        # print(f"document ques generated:-{self.document_ques_gen} and doc_ans:- {self.document_answer_gen}")

        llm_ques_gen_pipeline = ChatOpenAI(
                    model_name="opengvlab/internvl3-2b:free",  # OpenRouter model name
                    openai_api_base="https://openrouter.ai/api/v1",
                    openai_api_key=DEEPSEEK_API_KEY,
                    temperature=0.7,
                    max_tokens=1024
                )
        
        # 3. Configure the question generation chain
        ques_gen_chain = load_summarize_chain(
                        llm=llm_ques_gen_pipeline,
                        chain_type="refine",
                        verbose=False,
                        question_prompt=PROMPT_QUESTIONS,
                        refine_prompt=REFINE_PROMPT_QUESTIONS,
                        document_variable_name="context",
                        return_intermediate_steps=True
                )
        
        ques = ques_gen_chain.invoke(self.document_ques_gen)
    
        
        embedding= HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        vectorstore=FAISS.from_documents(documents=self.document_answer_gen,embedding=embedding)
        
        llm_ans_gen = ChatOpenAI(
                    model_name="opengvlab/internvl3-2b:free",  # OpenRouter model name
                    openai_api_base="https://openrouter.ai/api/v1",
                    openai_api_key=DEEPSEEK_API_KEY,
                    temperature=0.7,
                    max_tokens=1024)
        
        return ques_gen_chain,llm_ans_gen,vectorstore
    
    
    def generate_questions(self,gen_quen):
        try:
            result = gen_quen.invoke({"input_documents": self.document_ques_gen})
            
            # Extract outputs safely
            final_output = ""
            if "output_text" in result:
                final_output = result["output_text"]
            elif "output" in result:
                final_output = result["output"]
                
            intermediate_steps = result.get("intermediate_steps", [])
            
            # Process intermediate steps
            clean_steps = []
            for step in intermediate_steps:
                if isinstance(step, dict):
                    clean_steps.append(step.get("output_text", ""))
                else:
                    clean_steps.append(str(step))
            
            # Clean final output
            final_questions = [
                line.strip() 
                for line in final_output.split("\n") 
                if line.strip() and not line.startswith("QUESTIONS:")
            ]
            pattern = re.compile(r'\s\*{2}\w+\.:\*.?')
            
            final_questions=[pattern.sub("",line) for line in final_questions]
            # print(f"final questions:-{final_questions}")
            return final_questions
          
        except Exception as e:
            print(f"Generation Error: {str(e)}")
            return {"error": str(e)}
        
    def generate_answer(self,vectorstore):
        llm_ans_gen = ChatOpenAI(
                model_name="opengvlab/internvl3-2b:free",  # OpenRouter model name
                openai_api_base="https://openrouter.ai/api/v1",
                openai_api_key=DEEPSEEK_API_KEY,
                temperature=0.7,
                max_tokens=1024)
        
        answer_generation_chain = RetrievalQA.from_chain_type(llm=llm_ans_gen,
                                    chain_type="stuff", 
                                    retriever=vectorstore.as_retriever(),
                                    return_source_documents=True)
        
        return answer_generation_chain
        
    def final_result(self):
        
        ques_gen_chain, llm_ans_gen, vectorstore = self.get_document_quen_ans()
        questions = self.generate_questions(ques_gen_chain)
        answer = self.generate_answer(vectorstore)
        return answer, questions
    
    
    
path= r"D:\Vaibhav_PC\GenerativeAI\GI\Interview_QA\data\SDG.pdf"
# if __name__ == "__main__":
#     llm_pipeline = LLM_pipeline(path)
#     answer, question =llm_pipeline.final_result()
#     print(question)


    