import streamlit as st
import os
import fitz  # PyMuPDF
import re  # For regex in parser
import glob # For listing PDF files
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema.document import Document
import logging
import uuid

# --- Configuration ---
TRANSCRIPTS_DIR = "./expert_call_transcripts"
MODEL_DIR = "./models"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# === UPDATED FOR LLAMA 3.1 ===
LLM_MODEL_FILE = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" # Ensure this file is in ./models
# For 12GB VRAM with an 8B Q4_K_M model, you can try to offload many layers.
# -1 means all possible. Let's start with a high number; llama.cpp will adjust.
N_GPU_LAYERS = 35 # Adjust this based on your specific GPU and observed VRAM usage.
                  # If -1 causes issues, try a specific number like 32, 30, etc.
N_CTX = 8192      # Llama 3.1 can handle larger contexts; adjust based on RAM/performance
# ============================

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Custom PDF Parser for AlphaSense Format ---
def parse_alpha_sense_pdf_to_documents(pdf_path):
    if not os.path.exists(pdf_path):
        logging.error(f"Error: PDF file not found at {pdf_path}")
        return []
    doc = fitz.open(pdf_path)
    all_lc_documents = []
    filename = os.path.basename(pdf_path)
    timestamp_pattern = re.compile(r"(\d{2}:\d{2}:\d{2})")
    speaker_pattern = re.compile(r"^(Client|Expert)\s*(?:\d{2}:\d{2}:\d{2})?")
    section_summary_pattern = re.compile(r"^Summary$", re.IGNORECASE)
    section_toc_pattern = re.compile(r"^Table of Contents$", re.IGNORECASE)
    section_bio_pattern = re.compile(r"^Expert Bio$", re.IGNORECASE)
    section_employment_pattern = re.compile(r"^Employment History$", re.IGNORECASE)
    section_transcript_pattern = re.compile(r"^Interview Transcript$", re.IGNORECASE)
    current_section = "Header/Metadata"; current_speaker = None; current_timestamp = None
    current_accumulated_text_blocks = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num); page_dict = page.get_text("dict", sort=True)
        for block_item in page_dict["blocks"]:
            if block_item["type"] == 0: # Text block
                block_full_text_lines = []
                for line_item in block_item["lines"]:
                    line_text = "".join([span["text"] for span in line_item["spans"]])
                    block_full_text_lines.append(line_text)
                block_text_for_processing = "\n".join(block_full_text_lines).strip()
                if not block_text_for_processing: continue
                lines_from_block = block_text_for_processing.splitlines()
                for line_text in lines_from_block:
                    line_text = line_text.strip();
                    if not line_text: continue
                    new_section_name = None
                    if section_summary_pattern.match(line_text): new_section_name = "Summary"
                    elif section_toc_pattern.match(line_text): new_section_name = "Table of Contents"
                    elif section_bio_pattern.match(line_text): new_section_name = "Expert Bio"
                    elif section_employment_pattern.match(line_text): new_section_name = "Employment History"
                    elif section_transcript_pattern.match(line_text): new_section_name = "Interview Transcript"
                    if new_section_name:
                        if current_accumulated_text_blocks:
                            content = "\n".join(current_accumulated_text_blocks)
                            metadata = {"source_pdf": filename, "page": page_num + 1, "section": current_section, 
                                        "speaker": current_speaker if current_speaker else "SectionText", "timestamp": current_timestamp}
                            all_lc_documents.append(Document(page_content=content, metadata=metadata))
                        current_accumulated_text_blocks = []; current_section = new_section_name
                        current_speaker = None; current_timestamp = None; continue
                    if current_section == "Interview Transcript":
                        speaker_match = speaker_pattern.match(line_text)
                        if speaker_match:
                            if current_accumulated_text_blocks:
                                content = "\n".join(current_accumulated_text_blocks)
                                metadata = {"source_pdf": filename, "page": page_num + 1, "section": current_section, 
                                            "speaker": current_speaker, "timestamp": current_timestamp}
                                all_lc_documents.append(Document(page_content=content, metadata=metadata))
                            current_accumulated_text_blocks = []; current_speaker = speaker_match.group(1)
                            ts_in_line_search = timestamp_pattern.search(line_text)
                            if ts_in_line_search:
                                current_timestamp = ts_in_line_search.group(1); text_after_speaker_and_ts = line_text
                                # Clean the line more carefully
                                temp_text = line_text
                                if speaker_match.group(0): temp_text = temp_text.replace(speaker_match.group(0),"",1).strip()
                                if current_timestamp: temp_text = temp_text.replace(current_timestamp,"",1).strip()
                                text_after_speaker_and_ts = temp_text
                                if text_after_speaker_and_ts: current_accumulated_text_blocks.append(text_after_speaker_and_ts)
                            else:
                                current_timestamp = None; text_after_speaker = line_text[len(speaker_match.group(0)):].strip()
                                if text_after_speaker: current_accumulated_text_blocks.append(text_after_speaker)
                        elif current_speaker: # Continued text for current speaker
                            if timestamp_pattern.fullmatch(line_text) and current_timestamp == line_text: pass # Avoid duplicate timestamp
                            else: current_accumulated_text_blocks.append(line_text)
                        else: current_accumulated_text_blocks.append(line_text) # Text before first speaker
                    else: # Non-transcript sections
                        current_accumulated_text_blocks.append(line_text)
                        # current_speaker should ideally be None or "SectionText" here
                        # current_speaker = "SectionText" # if you want to explicitly mark it
                        current_timestamp = None
    if current_accumulated_text_blocks: # Add any last accumulated block
        content = "\n".join(current_accumulated_text_blocks)
        metadata = {"source_pdf": filename, "page": doc.page_count, "section": current_section,
                    "speaker": current_speaker if current_speaker else "SectionText", "timestamp": current_timestamp}
        all_lc_documents.append(Document(page_content=content, metadata=metadata))
    doc.close(); return all_lc_documents

# --- Caching for expensive operations ---
@st.cache_resource
def load_embedding_model(model_name_or_path_segment):
    local_model_full_path = os.path.join(MODEL_DIR, model_name_or_path_segment)
    model_to_load_from = local_model_full_path if os.path.isdir(local_model_full_path) else model_name_or_path_segment
    try:
        if model_to_load_from == local_model_full_path: pass
        else: st.warning(f"Local path {local_model_full_path} not found. Attempting to load '{model_name_or_path_segment}' (may need internet).")
        embeddings = SentenceTransformerEmbeddings(model_name=model_to_load_from, model_kwargs={'device': 'cpu'})
        st.success(f"Embedding model '{model_name_or_path_segment}' loaded.")
        return embeddings
    except Exception as e:
        st.error(f"Error loading embedding model '{model_name_or_path_segment}': {e}")
        logging.exception("Embedding model loading error:")
        return None

def create_combined_vector_store(transcript_dir_path, _embedding_model):
    all_structured_documents_from_all_pdfs = []
    if not os.path.isdir(transcript_dir_path):
        st.error(f"Transcripts directory not found: {transcript_dir_path}")
        return None, 0
    pdf_files = sorted(glob.glob(os.path.join(transcript_dir_path, "*.pdf")))
    if not pdf_files:
        st.warning(f"No PDF files found in {transcript_dir_path}.")
        return None, 0
    st.info(f"Found {len(pdf_files)} PDF(s) to process...")
    progress_text_area = st.empty(); progress_bar = st.progress(0)
    for i, pdf_path in enumerate(pdf_files):
        filename_display = os.path.basename(pdf_path)
        progress_text_area.info(f"Parsing ({i+1}/{len(pdf_files)}): {filename_display}")
        logging.info(f"Parsing: {filename_display}")
        parsed_docs_single = parse_alpha_sense_pdf_to_documents(pdf_path)
        if parsed_docs_single: all_structured_documents_from_all_pdfs.extend(parsed_docs_single)
        progress_bar.progress((i + 1) / len(pdf_files))
    progress_text_area.info(f"Total initial segments from all PDFs: {len(all_structured_documents_from_all_pdfs)}")
    if not all_structured_documents_from_all_pdfs:
        st.error("No documents could be extracted from any PDF files.")
        return None, 0
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    all_splits = text_splitter.split_documents(all_structured_documents_from_all_pdfs)
    if not all_splits:
        st.error("Text splitting of combined documents resulted in no final chunks.")
        return None, len(pdf_files)
    progress_text_area.info(f"Creating combined vector store from {len(all_splits)} final text chunks...")
    try:
        vector_store = FAISS.from_documents(documents=all_splits, embedding=_embedding_model)
        progress_text_area.success(f"Combined vector store created! {len(pdf_files)} transcript(s) included.")
        return vector_store, len(pdf_files)
    except Exception as e:
        progress_text_area.error(f"Error creating combined vector store: {e}")
        logging.exception("Combined vector store creation error:")
        return None, len(pdf_files)

@st.cache_resource
def load_llm(model_file, n_gpu_layers, n_ctx):
    llm_path = os.path.join(MODEL_DIR, model_file)
    if not os.path.exists(llm_path):
        st.error(f"LLM file not found at {llm_path}")
        return None
    try:
        llm = LlamaCpp(
            model_path=llm_path,
            n_gpu_layers=n_gpu_layers,
            n_batch=512, 
            n_ctx=n_ctx,
            verbose=False, # Set to True for Llama.cpp init details
            temperature=0.1, 
            # chat_format="llama-3" # Optional: try if your GGUF/llama-cpp-python supports it well.
                                  # If it causes issues, remove it and rely on manual templating.
        )
        st.success("LLM loaded successfully.")
        return llm
    except Exception as e:
        st.error(f"Error loading LLM: {e}")
        logging.exception("LLM loading error:")
        return None

# --- Prompts for Llama 3.1 Instruct ---
CONDENSE_QUESTION_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant that rephrases a follow-up question to be a standalone question,
based on a provided chat history. Do not answer the question, just rephrase it.
If the follow up input is already a standalone question, just return it as is.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Chat History:
{chat_history}

Follow Up Input: {question}
Standalone question:<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(CONDENSE_QUESTION_TEMPLATE)

QA_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an AI assistant for answering questions based on provided expert call transcripts.
Use ONLY the following pieces of context (relevant excerpts from transcripts) to answer the question at the end.
If the context does not contain the answer, state "The provided transcripts do not contain specific information on this topic."
Do not make up information or use external knowledge.
Provide a concise answer. Refer to information from the context.
The user will be shown the source documents (including filename, page, speaker, and timestamp) separately.
Answer the question directly based on the information found.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Context from transcripts:
---------------------
{context}
---------------------
Question: {question}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Helpful Answer:
"""
QA_PROMPT = PromptTemplate(template=QA_TEMPLATE, input_variables=["context", "question"])

# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="Multi-Transcript Expert Call Q&A (Llama 3.1)")
st.title("Multi-Transcript Expert Call Q&A 💬 (using Llama 3.1)")

if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "display_messages" not in st.session_state: st.session_state.display_messages = []
if "vector_store" not in st.session_state: st.session_state.vector_store = None
if "processed_files_count" not in st.session_state: st.session_state.processed_files_count = 0
if "expanded_source_key" not in st.session_state: st.session_state.expanded_source_key = None

# --- Main Application Logic ---
if not os.path.isdir(MODEL_DIR):
    st.error(f"Models directory '{MODEL_DIR}' not found. Please create it and add your models.")
    st.stop()

embedding_model = load_embedding_model(EMBEDDING_MODEL_NAME)
llm = load_llm(LLM_MODEL_FILE, N_GPU_LAYERS, N_CTX)

# --- Sidebar ---
st.sidebar.header("Knowledge Base Setup")
if not os.path.exists(TRANSCRIPTS_DIR):
    os.makedirs(TRANSCRIPTS_DIR)
    st.sidebar.warning(f"Created directory: {TRANSCRIPTS_DIR}. Please add PDF files and click 'Load/Refresh'.")
try:
    pdf_files_in_dir = glob.glob(os.path.join(TRANSCRIPTS_DIR, "*.pdf"))
    st.sidebar.caption(f"{len(pdf_files_in_dir)} PDF(s) found in directory.")
except Exception as e: st.sidebar.caption(f"Could not access directory: {e}")

if st.sidebar.button("Load/Refresh All Transcripts Knowledge Base"):
    st.session_state.vector_store = None; st.session_state.processed_files_count = 0
    st.session_state.chat_history = []; st.session_state.display_messages = []
    st.session_state.expanded_source_key = None
    if embedding_model and llm:
        vs, count = create_combined_vector_store(TRANSCRIPTS_DIR, embedding_model)
        st.session_state.vector_store = vs; st.session_state.processed_files_count = count
        st.rerun() 
    else: st.sidebar.error("Core models (Embeddings/LLM) not loaded. Cannot process transcripts.")

if st.session_state.vector_store:
    st.sidebar.success(f"KB Ready: {st.session_state.processed_files_count} transcript(s) loaded.")
else: st.sidebar.warning("Knowledge base not loaded. Click 'Load/Refresh' button.")

if st.sidebar.button("Clear Chat History"):
    st.session_state.chat_history = []; st.session_state.display_messages = []
    st.session_state.expanded_source_key = None
    st.sidebar.success("Chat history cleared."); st.rerun()

# --- ConversationalRetrievalChain Setup and Chat UI ---
if st.session_state.vector_store and llm and embedding_model:
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')
    for q, a in st.session_state.chat_history: memory.save_context({"question": q}, {"answer": a})

    simple_document_prompt_template = "--- Retrieved Document Snippet ---\n{page_content}\n--- End Snippet ---"
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=st.session_state.vector_store.as_retriever(search_kwargs={"k": 5}),
        memory=memory, return_source_documents=True,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={
            "prompt": QA_PROMPT,
            "document_prompt": PromptTemplate.from_template(simple_document_prompt_template),
            "document_variable_name": "context"
        },
    )
    st.header(f"Chat across {st.session_state.processed_files_count} Loaded Transcript(s) (Llama 3.1)")

    for msg_idx, msg in enumerate(st.session_state.display_messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("View Sources for this Answer"):
                    for i, doc_source in enumerate(msg["sources"]):
                        meta = doc_source.metadata
                        source_desc_parts = [f"Source Doc {i+1} from '{meta.get('source_pdf', 'Unknown PDF')}'",
                                             f"Page {meta.get('page', 'N/A')}"]
                        if meta.get('section') and meta.get('section') != 'Header/Metadata':
                            source_desc_parts.append(f"Section: {meta.get('section')}")
                        if meta.get('speaker') and meta.get('speaker') != 'SectionText':
                             source_desc_parts.append(f"Speaker: {meta.get('speaker')}")
                        if meta.get('timestamp'): source_desc_parts.append(f"({meta.get('timestamp')})")
                        source_desc = " - ".join(source_desc_parts)
                        st.caption(source_desc)
                        st.markdown(f"> {doc_source.page_content[:350].strip()}...")
                        unique_source_key = f"src_toggle_{msg.get('id', msg_idx)}_{i}"
                        if st.button(f"Show/Hide Full Text Source {i+1}", key=unique_source_key):
                            if st.session_state.expanded_source_key == unique_source_key:
                                st.session_state.expanded_source_key = None
                            else: st.session_state.expanded_source_key = unique_source_key
                            st.rerun()
                        if st.session_state.expanded_source_key == unique_source_key:
                            st.text_area(f"Full text: {source_desc}", doc_source.page_content, 
                                         height=300, key=f"text_area_{unique_source_key}")
    
    if user_question := st.chat_input(f"Ask across all loaded transcripts (Llama 3.1):"):
        if not st.session_state.vector_store :
            st.warning("Knowledge base is not loaded. Please click 'Load/Refresh All Transcripts' in the sidebar.")
        else:
            msg_id_base = str(uuid.uuid4())
            user_msg_id = f"user_{msg_id_base}"
            st.session_state.display_messages.append({"role": "user", "content": user_question, "id": user_msg_id})
            with st.chat_message("user"): st.markdown(user_question)
            with st.spinner("Searching across transcripts with Llama 3.1..."):
                try:
                    result = qa_chain.invoke({"question": user_question})
                    answer = result["answer"]
                    if "<|eot_id|>" in answer:
                        answer = answer.split("<|eot_id|>")[0].strip()
                    source_documents = result.get("source_documents", [])
                    ai_msg_id = f"ai_{msg_id_base}"
                    st.session_state.display_messages.append({
                        "role": "assistant", "content": answer,
                        "sources": source_documents, "id": ai_msg_id
                    })
                    st.session_state.chat_history.append((user_question, answer))
                    st.rerun()
                except Exception as e:
                    st.error(f"Error during Q&A: {e}")
                    logging.exception("Q&A processing error:")
                    error_msg_id = f"ai_error_{msg_id_base}"
                    st.session_state.display_messages.append({
                        "role": "assistant", "content": f"Sorry, an error occurred: {str(e)[:200]}...",
                        "id": error_msg_id
                    })
                    st.rerun()
else:
    st.info("Welcome! Please click 'Load/Refresh All Transcripts Knowledge Base' in the sidebar to begin.")
    if not embedding_model: st.error("Embedding model failed to load. Check console for errors.")
    if not llm: st.error("LLM (Llama 3.1) failed to load. Check console for errors or model path.")
    elif not st.session_state.vector_store and (embedding_model and llm): pass

st.markdown("---")
st.markdown("Multi-Transcript AlphaSense RAG Q&A Tool (Llama 3.1)")