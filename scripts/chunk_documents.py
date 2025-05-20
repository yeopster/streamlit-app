from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def chunk_documents(input_dir, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = []
    for txt_file in os.listdir(input_dir):
        if txt_file.endswith(".txt"):
            with open(os.path.join(input_dir, txt_file), "r", encoding="utf-8") as f:
                text = f.read()
            doc_chunks = splitter.split_text(text)
            chunks.extend([{"text": chunk, "source": txt_file} for chunk in doc_chunks])
    return chunks

if __name__ == "__main__":
    chunks = chunk_documents("data/processed/")
    print(f"Created {len(chunks)} chunks")
    # Save chunks for inspection (optional)
    with open("data/chunks.txt", "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(f"Chunk {i+1} (Source: {chunk['source']}):\n{chunk['text']}\n\n")