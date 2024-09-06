import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to set up a conversational AI chain
def get_conversational_chain():
    prompt_template = """
    You are an advanced AI system designed to provide detailed information about government schemes based on the data provided in multiple text documents. Your goal is to assist users by answering their questions about various government schemes, using the context and information from these documents.

When asked about government schemes in a specific domain, follow these steps:

1. Extract and summarize the types of schemes available in the specified domain (e.g., healthcare, education, employment, social security, financial aid).
2. For each type of scheme, provide detailed information including:
   - Key features (e.g., eligibility criteria, benefits, application process).
   - Advantages and disadvantages.
   - Specific requirements or considerations.
3. If detailed information about a specific scheme is not present in the provided context, respond with a general description of common features and aspects of such schemes. Clearly state that the information is generalized due to the absence of specific data.
4. If the general description is not sufficient or if the domain is not well-defined, respond with, "The information is not available in the context."

Ensure that your response is accurate and comprehensive, based on the data in the context, or provide a well-informed general response when specifics are lacking. Avoid speculative answers.

Context:\n{context}\n
Domain: {domain}\n

Please list all available government schemes in the given domain with detailed information or a generalized description if specific details are missing.
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "domain"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain



# Function to process user input and retrieve relevant answers
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "domain": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

# Function to get text from .txt files
def get_text_from_files(file_paths):
    text = ""
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            text += file.read() + "\n"
    return text

# Main function for the app
def main():
    st.set_page_config(page_title="Mahithi")
    st.title("Mahithi : A Government Cognitive Assistant")

    # Create a dropdown menu for navigation
    nav = st.selectbox("Navigate to", ["Home", "About", "Agriculture", "Education", "Personal", "Marriage"])

    # Create sections for each navigation option
    if nav == "Home":
        st.header("Welcome to Mahithi!")
        st.write("Explore various government schemes and services that cater to your needs.")
        st.subheader("File Processing and Question Answering")

        # Predefined directory containing text files
        text_files_directory = './data'
        
        # List all text files in the directory
        text_file_paths = [os.path.join(text_files_directory, file) for file in os.listdir(text_files_directory) if file.endswith('.txt')]

        if not text_file_paths:
            st.error("No text files found in the specified directory.")
            return

        # Process files and create vector store
        with st.spinner("Processing files..."):
            raw_text = get_text_from_files(text_file_paths)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
        st.success("File processing completed")

        # Allow user to ask questions
        user_question = st.text_input("Ask a Question from the Text Files")

        if user_question:
            user_input(user_question)

    elif nav == "About":
        st.header("About Mahithi")
        st.write("""
            Mahithi is a government cognitive assistant designed to help citizens easily find and access various government schemes. 
            Whether you are looking for support in agriculture, education, personal services, or marriage-related information, 
            Mahithi aims to simplify the process of discovering relevant schemes and services tailored to your needs. 
            With Mahithi, solving problems related to finding the right schemes has never been easier.
        """)

    elif nav == "Agriculture":
        st.header("Agriculture Information")
        with st.expander("Overview of Agricultural Schemes"):
            st.write("""
                This section provides a comprehensive list of government schemes available for farmers, 
                covering subsidies, loans, insurance, and support for sustainable farming practices.
            """)
        with st.expander("Subsidies and Financial Support"):
            st.write("""
                Explore schemes that offer financial support for purchasing seeds, fertilizers, and farming equipment. 
                Find out how to apply and what eligibility criteria need to be met.
            """)
        with st.expander("Sustainable Farming Initiatives"):
            st.write("""
                Learn about government initiatives that promote sustainable and organic farming, 
                including training programs, grants, and certification processes.
            """)

    elif nav == "Education":
        st.header("Education Information")
        with st.expander("Educational Grants and Scholarships"):
            st.write("""
                Discover a variety of scholarships and grants available for students at different levels, 
                from primary education to higher studies, including specialized programs for underprivileged communities.
            """)
        with st.expander("Skill Development Programs"):
            st.write("""
                Access information about skill development programs designed to enhance employability 
                and vocational skills, focusing on both rural and urban populations.
            """)
        with st.expander("School and College Infrastructure Support"):
            st.write("""
                Explore schemes aimed at improving educational infrastructure, 
                including funding for new schools, digital classrooms, and educational resources.
            """)

    elif nav == "Personal":
        st.header("Personal Information")
        with st.expander("Identity and Documentation"):
            st.write("""
                Learn about the process of obtaining essential documents like Aadhar cards, 
                voter IDs, and birth certificates, along with government services that assist with these processes.
            """)
        with st.expander("Healthcare and Insurance"):
            st.write("""
                Find out about government healthcare schemes and insurance policies that provide coverage for medical expenses, 
                including specific programs for senior citizens, women, and children.
            """)
        with st.expander("Social Security and Welfare"):
            st.write("""
                Understand the various social security and welfare programs available for different demographics, 
                including pensions, disability benefits, and unemployment support.
            """)

    elif nav == "Marriage":
        st.header("Marriage Information")
        with st.expander("Marriage Registration"):
            st.write("""
                Get details on how to register a marriage, including the necessary documents, 
                the application process, and where to apply.
            """)
        with st.expander("Government Marriage Schemes"):
            st.write("""
                Explore government schemes that provide financial assistance for marriages, 
                especially for those belonging to economically weaker sections and marginalized communities.
            """)
        with st.expander("Family and Child Welfare"):
            st.write("""
                Learn about government programs aimed at supporting families, 
                including maternity benefits, child welfare schemes, and family counseling services.
            """)

if __name__ == "__main__":
    main()
