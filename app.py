import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from PyPDF2 import PdfReader
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from openai import OpenAI

# ✅ OpenRouter Credentials
client = OpenAI(
    api_key="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",
    base_url="https://openrouter.ai/api/v1"
)

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

# ✅ Custom LLM Wrapper for OpenRouter
class OpenRouterLLM(LLM):
    def _call(self, prompt: str, stop=None) -> str:
        completion = client.chat.completions.create(
            model="qwen/qwen3-coder:free",
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content

    @property
    def _identifying_params(self):
        return {"name": "OpenRouterLLM"}

    @property
    def _llm_type(self):
        return "custom"

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

    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenRouterLLM(),
        retriever=db.as_retriever()
    )

    query = st.text_input("Ask a policy-related question:")
    if query:
        answer = qa_chain.run(query)
        st.subheader("Answer:")
        st.write(answer)

        email_prompt = f"Write a short professional email to HR based on this policy answer: '{answer}'."
        email_completion = client.chat.completions.create(
            model="qwen/qwen3-coder:free",
            messages=[{"role": "user", "content": email_prompt}]
        )
        email_text = email_completion.choices[0].message.content

        st.subheader("Generated Email:")
        st.text_area("Email Draft", email_text, height=200)

        if st.button("📧 Send Email to HR"):
            send_email("hrteam@company.com", "Policy Query Request", email_text)
            st.success("✅ Email sent successfully!")
