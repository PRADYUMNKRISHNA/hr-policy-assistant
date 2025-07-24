import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from PyPDF2 import PdfReader
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

# ✅ Fixed Credentials (OpenRouter + Gmail)
os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

EMAIL_USER = "pradyumnkrishna0@gmail.com"
EMAIL_PASS = "nxhdpxtspbeyedls"

# ✅ Email Sending Function
def send_email(to_email, subject, body):
    msg = MIMEMultipart()
    msg["From"] = EMAIL_USER
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(EMAIL_USER, EMAIL_PASS)
    server.sendmail(EMAIL_USER, to_email, msg.as_string())
    server.quit()

# ✅ Streamlit App UI
st.title("📄 HR Policy Assistant (Gen AI + Email Automation)")
uploaded_file = st.file_uploader("Upload your HR Policy PDF", type=["pdf"])

if uploaded_file:
    pdf_reader = PdfReader(uploaded_file)
    policy_text = "".join([page.extract_text() for page in pdf_reader.pages])

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_text(policy_text)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_texts(texts, embeddings)

    # ✅ FIX: Added API key and base explicitly
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(
            model="qwen/qwen3-coder:free",
            temperature=0,
            openai_api_key=os.environ["OPENAI_API_KEY"],
            openai_api_base=os.environ["OPENAI_API_BASE"]
        ),
        retriever=db.as_retriever()
    )

    query = st.text_input("Ask a policy-related question:")
    if query:
        answer = qa_chain.run(query)
        st.subheader("Answer:")
        st.write(answer)

        email_prompt = f"Write a short professional email to HR based on this policy answer: '{answer}'."
        email_text = ChatOpenAI(
            model="qwen/qwen3-coder:free",
            temperature=0.3,
            openai_api_key=os.environ["OPENAI_API_KEY"],
            openai_api_base=os.environ["OPENAI_API_BASE"]
        ).predict(email_prompt)

        st.subheader("Generated Email:")
        st.text_area("Email Draft", email_text, height=200)

        if st.button("📧 Send Email to HR"):
            send_email("hrteam@company.com", "Policy Query Request", email_text)
            st.success("✅ Email sent successfully!")
