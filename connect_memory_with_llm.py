import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredImageLoader
from langchain_core.documents import Document
import ollama
import io
from PIL import Image
import pytesseract

def extract_text_from_image(image_path: str) -> str:
    try:
        with open(image_path, "rb") as file:
            file_bytes = file.read()

        image = Image.open(io.BytesIO(file_bytes))

        # Convert to grayscale (improves OCR)
        gray_image = image.convert("L")

        extracted_text = pytesseract.image_to_string(gray_image).strip()

        return extracted_text

    except Exception as e:
        print(f"Error extracting text: {str(e)}")
        raise


# taking user learning preference as input
learning_style = input(
    "Choose learning style (bullets / short notes / detailed notes / examples): "
)

# Load documents from uploaded files/User input
user_docs = []
UPLOAD_DIR = "uploaded_files"

for filename in os.listdir(UPLOAD_DIR):
    file_path = os.path.join(UPLOAD_DIR, filename)

    # ---- PDF FLOW ----
    if filename.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        user_docs.extend(loader.load())

    # ---- IMAGE FLOW ----
    elif filename.lower().endswith((".png", ".jpg", ".jpeg")):
        extracted_text = extract_text_from_image(file_path)

        if extracted_text.strip():
            user_docs.append(
                Document(
                    page_content=extracted_text,
                    metadata={"source": filename}
                )
            )

    else:
        continue

user_report_text = "\n".join(doc.page_content for doc in user_docs)

# Load pre-existing medical knowledge vector store
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    "vectorstore/db_faiss",   
    embeddings,
    allow_dangerous_deserialization=True
)

# Semantic search to retrieve relevant medical knowledge
semantic_query = user_report_text[:3000]

retrieved_medical_docs = vectorstore.similarity_search(
    semantic_query,
    k=8
)

medical_context = "\n".join(
    doc.page_content for doc in retrieved_medical_docs
)

# Construct the final prompt with user report and retrieved medical knowledge
final_context = f"""
USER UPLOADED CONTENT:
{user_report_text}

RELATED MEDICAL KNOWLEDGE:
{medical_context}
"""

SUMMARY_PROMPT = """
You are an AI Knowledge Distiller that creates high-quality revision notes.

Strict rules:
- Use ONLY the provided context.
- Do NOT hallucinate or add external knowledge.
- Remove duplicated or repeated information.
- Merge all documents into one coherent understanding.
- Identify and prioritize the most important concepts.
- Ignore irrelevant, redundant, or low-value details.
- If the information is incomplete or unclear, say so explicitly.

User learning preference:
{learning_style}

Context (combined text extracted from multiple uploaded documents):
{context}

Task:
Generate a SINGLE, clean, one-page revision sheet.

The output must:
- Be concise, structured, and easy to revise
- Include key concepts, definitions, and important points
- Use simple, clear explanations
- Include examples ONLY if they improve understanding
- Follow the user’s learning preference in style and format
  (e.g., bullet points, short notes, detailed notes, exam-focused)

Start directly with the revision content.
"""

prompt = PromptTemplate(
    template=SUMMARY_PROMPT,
    input_variables=["context", "learning_style"]
)

final_prompt = prompt.format(context=final_context, learning_style=learning_style)

# Generate summary using Ollama LLM
response = ollama.chat(
    model="MedAIBase/MedGemma1.5:4b",
    messages=[
        {"role": "system", "content": "You are a careful medical AI assistant."},
        {"role": "user", "content": final_prompt}
    ]
)

print(response["message"]["content"])




