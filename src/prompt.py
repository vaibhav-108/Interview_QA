from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate



PROMPT_QUESTIONS = ChatPromptTemplate.from_messages([
    ("human", """
    You are an expert at creating questions based on coding materials. 
    **Generate questions in ENGLISH ONLY** for programming exams.
    
    Text: {context}
    
    English Questions:
    """)
])

REFINE_PROMPT_QUESTIONS = ChatPromptTemplate.from_messages([
    ("human", """
    Refine these questions in ENGLISH ONLY:
    1. Keep original numbering
    2. Add [NEW] for new questions
    3. Combine overlaps
    
    Existing Questions:
    {existing_answer}
    
    New Context:
    {context}
    
    Refined English Questions:
    """)
])