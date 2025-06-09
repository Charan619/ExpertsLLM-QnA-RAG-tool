import fitz  # PyMuPDF
import re
import os
from langchain.schema.document import Document # Or from langchain_core.documents import Document

def parse_alpha_sense_pdf_to_documents(pdf_path):
    """
    Parses a Tegus Expert Insights PDF by AlphaSense into a list of
    LangChain Document objects, trying to segment by speaker turns
    and sections.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        list[Document]: A list of LangChain Document objects.
    """
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return []

    doc = fitz.open(pdf_path)
    all_lc_documents = []
    filename = os.path.basename(pdf_path)

    # --- Regex patterns ---
    # Timestamps like 00:00:00
    timestamp_pattern = re.compile(r"(\d{2}:\d{2}:\d{2})")
    # Speaker turns (Client or Expert) possibly followed by a timestamp
    speaker_pattern = re.compile(r"^(Client|Expert)\s*(?:\d{2}:\d{2}:\d{2})?")
    # Section headers (simple check, can be made more robust)
    section_summary_pattern = re.compile(r"^Summary$", re.IGNORECASE)
    section_toc_pattern = re.compile(r"^Table of Contents$", re.IGNORECASE)
    section_bio_pattern = re.compile(r"^Expert Bio$", re.IGNORECASE)
    section_employment_pattern = re.compile(r"^Employment History$", re.IGNORECASE)
    section_transcript_pattern = re.compile(r"^Interview Transcript$", re.IGNORECASE)

    # Global metadata (attempt to extract from first page)
    # This part would need more robust parsing based on layout if needed,
    # for now, we'll focus on the transcript content.
    # You could use text block coordinates if the layout is very consistent.
    global_metadata = {"source_pdf": filename}
    # Example: (Not robustly implemented here, just a placeholder)
    # first_page_text = doc[0].get_text("text")
    # if "DATE PUBLISHED" in first_page_text:
    #     global_metadata["date_published"] = "..."


    current_section = "Header/Metadata" # Initial section
    current_speaker = None
    current_timestamp = None
    current_speaker_text_blocks = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("blocks", sort=True) # Get text blocks with coordinates

        for i, block in enumerate(blocks):
            block_text = block[4].strip() # The text content of the block
            if not block_text:
                continue

            # --- Section Detection ---
            is_new_section = False
            if section_summary_pattern.search(block_text):
                current_section = "Summary"
                is_new_section = True
            elif section_toc_pattern.search(block_text):
                current_section = "Table of Contents"
                is_new_section = True
            elif section_bio_pattern.search(block_text):
                current_section = "Expert Bio"
                is_new_section = True
            elif section_employment_pattern.search(block_text):
                current_section = "Employment History"
                is_new_section = True
            elif section_transcript_pattern.search(block_text):
                current_section = "Interview Transcript"
                is_new_section = True
            
            # If it was a section header, store previous speaker's text and skip this block
            if is_new_section:
                if current_speaker and current_speaker_text_blocks:
                    content = "\n".join(current_speaker_text_blocks)
                    metadata = {
                        "source_pdf": filename,
                        "page": page_num + 1,
                        "section": "Interview Transcript", # Assume previous was transcript
                        "speaker": current_speaker,
                        "timestamp": current_timestamp
                    }
                    all_lc_documents.append(Document(page_content=content, metadata=metadata))
                    current_speaker_text_blocks = []
                    current_speaker = None # Reset speaker
                
                # Store the section header itself if desired, or just use for context
                # metadata = {"source_pdf": filename, "page": page_num + 1, "section_header": current_section}
                # all_lc_documents.append(Document(page_content=block_text, metadata=metadata))
                continue # Move to next block after processing section header

            # --- Transcript Processing ---
            if current_section == "Interview Transcript":
                speaker_match = speaker_pattern.match(block_text)
                timestamp_match_in_block = timestamp_pattern.search(block_text)

                if speaker_match:
                    # Save previous speaker's text
                    if current_speaker and current_speaker_text_blocks:
                        content = "\n".join(current_speaker_text_blocks)
                        metadata = {
                            "source_pdf": filename,
                            "page": page_num + 1, # Page where this segment *ends* or mostly is
                            "section": current_section,
                            "speaker": current_speaker,
                            "timestamp": current_timestamp
                        }
                        all_lc_documents.append(Document(page_content=content, metadata=metadata))
                        current_speaker_text_blocks = []

                    current_speaker = speaker_match.group(1) # Client or Expert
                    
                    # Extract timestamp if present with the speaker line
                    # The timestamp is often *part* of the speaker line or immediately after
                    if timestamp_match_in_block and block_text.startswith(current_speaker):
                         # Check if timestamp is part of the speaker line itself
                        ts_search_str = block_text[len(current_speaker):].strip()
                        ts_match = timestamp_pattern.match(ts_search_str)
                        if ts_match:
                            current_timestamp = ts_match.group(1)
                            # Remove speaker and timestamp from this block's text to avoid duplication
                            block_text_cleaned = ts_search_str[len(current_timestamp):].strip()
                            if block_text_cleaned:
                                current_speaker_text_blocks.append(block_text_cleaned)
                        else: # Timestamp not directly with speaker, might be on next line or this block is just speaker
                            current_timestamp = None # Reset if no new timestamp found
                            # The block might just be "Client" or "Expert"
                            # Add the rest of the block if any (after "Client " or "Expert ")
                            remaining_text = block_text[len(current_speaker):].strip()
                            if remaining_text: # This could be the start of their speech
                                current_speaker_text_blocks.append(remaining_text)
                    else: # No timestamp on the speaker line itself
                        current_timestamp = None
                        remaining_text = block_text[len(current_speaker):].strip()
                        if remaining_text:
                            current_speaker_text_blocks.append(remaining_text)
                
                elif current_speaker: # Continued text for the current speaker
                    # Check if this line IS a timestamp (sometimes on its own line after speaker)
                    if timestamp_pattern.fullmatch(block_text) and not current_timestamp:
                        current_timestamp = block_text
                        # Don't add timestamp itself as content
                    else:
                        current_speaker_text_blocks.append(block_text)
                else: # Text before the first speaker in "Interview Transcript" or other sections
                    # For sections like Summary, Bio, etc.
                    metadata = {
                        "source_pdf": filename,
                        "page": page_num + 1,
                        "section": current_section,
                    }
                    # We can accumulate these or create one doc per block
                    # For simplicity here, one doc per block for non-transcript text
                    all_lc_documents.append(Document(page_content=block_text, metadata=metadata))

            else: # Handling for Summary, Bio, TOC, Employment History
                  # We'll create one Document per text block for these sections for now.
                  # More sophisticated chunking can be applied later.
                metadata = {
                    "source_pdf": filename,
                    "page": page_num + 1,
                    "section": current_section,
                }
                all_lc_documents.append(Document(page_content=block_text, metadata=metadata))

    # Add any remaining text from the last speaker
    if current_speaker and current_speaker_text_blocks:
        content = "\n".join(current_speaker_text_blocks)
        metadata = {
            "source_pdf": filename,
            "page": doc.page_count, # Assume last page
            "section": current_section, # Should be Interview Transcript if speaker is active
            "speaker": current_speaker,
            "timestamp": current_timestamp
        }
        all_lc_documents.append(Document(page_content=content, metadata=metadata))

    doc.close()
    return all_lc_documents

def main():
    # Create a dummy PDF for testing that resembles the structure
    # For a real run, you'd replace 'dummy_expert_call.pdf' with your actual PDF path
    pdf_file_to_parse = "expert_calls.pdf" # Replace with your actual PDF file path

    # Check if the example PDF exists (you provided image, not file)
    if not os.path.exists(pdf_file_to_parse):
        print(f"Test PDF '{pdf_file_to_parse}' not found. Please provide a valid PDF.")
        print("This script will demonstrate parsing if a PDF is available.")
        # Create a very simple dummy PDF if you want to test the flow without a real AlphaSense PDF
        # This part is optional and for local testing if you don't have the PDF handy for the script
        if not os.path.exists("dummy_alphasense.pdf"):
            try:
                from reportlab.pdfgen import canvas
                from reportlab.lib.pagesizes import letter
                c = canvas.Canvas("dummy_alphasense.pdf", pagesize=letter)
                c.drawString(72, 750, "Summary")
                c.drawString(72, 730, "This is the summary text.")
                c.showPage()
                c.drawString(72, 750, "Interview Transcript")
                c.drawString(72, 730, "Client 00:00:00")
                c.drawString(72, 710, "This is the client's first question.")
                c.drawString(72, 690, "Expert 00:00:15")
                c.drawString(72, 670, "This is the expert's answer.")
                c.save()
                print("Created a dummy_alphasense.pdf for demonstration.")
                pdf_file_to_parse = "dummy_alphasense.pdf"
            except ImportError:
                print("reportlab not installed. Cannot create dummy PDF.")
                return
        else:
             pdf_file_to_parse = "dummy_alphasense.pdf"


    if os.path.exists(pdf_file_to_parse):
        print(f"Parsing PDF: {pdf_file_to_parse}")
        parsed_documents = parse_alpha_sense_pdf_to_documents(pdf_file_to_parse)

        if parsed_documents:
            print(f"\nSuccessfully parsed {len(parsed_documents)} LangChain Documents.")
            for i, lc_doc in enumerate(parsed_documents[:50]): # Print first 5
                print(f"\n--- Document {i+1} ---")
                print(f"Content: {lc_doc.page_content}...") # Print snippet
                print(f"Metadata: {lc_doc.metadata}")
            
            # Now you can use `parsed_documents` with your RAG system's
            # text splitter and vector store creation logic.
            # Example:
            # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            # all_splits = text_splitter.split_documents(parsed_documents)
            # vector_store = FAISS.from_documents(documents=all_splits, embedding=your_embedding_model)
            # print(f"\nSplit into {len(all_splits)} chunks for vector store.")

        else:
            print("No documents were parsed from the PDF.")
    else:
        print(f"Could not find or create a PDF to parse: {pdf_file_to_parse}")


if __name__ == "__main__":
    main()