import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

#this is to make a separate prompt for each chunck, based on the source
def build_prompt(query: str, retrieved_chunks: list) -> str:
    
    context = "\n\n".join(
        f"[{i}] (source: {chunk.source_id}, section: {chunk.section})\n{chunk.text}"
        for i, (chunk, _) in enumerate(retrieved_chunks)
    )

    return f"""Answer the question using ONLY the numbered sources below.
After each factual claim, cite the source number in brackets, e.g. [0], [1].
If the answer is not contained in the sources, respond with:
"Not found in provided sources."

Sources:
{context}

Question: {query}
Answer:"""


def _generate_with_ollama(prompt: str) -> str:
    import ollama
    response = ollama.chat(
        model=config.OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2},
    )
    return response["message"]["content"]


def _generate_with_nim(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(base_url=config.NIM_BASE_URL, api_key=config.NIM_API_KEY)
    response = client.chat.completions.create(
        model=config.NIM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=500,
    )
    return response.choices[0].message.content


def generate_answer(prompt: str) -> str:
    
    if config.GENERATION_BACKEND == "ollama":
        return _generate_with_ollama(prompt)
    elif config.GENERATION_BACKEND == "nim":
        return _generate_with_nim(prompt)
    else:
        raise ValueError(f"Unknown GENERATION_BACKEND: {config.GENERATION_BACKEND}")


def extract_citations(answer: str, num_sources: int) -> dict:
    
    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
    citation_map = {}

    for i, sentence in enumerate(sentences):
        matches = re.findall(r'\[(\d+)\]', sentence)
        if matches:
            src_idx = int(matches[0])
            if 0 <= src_idx < num_sources:
                citation_map[i] = src_idx

    return sentences, citation_map


if __name__ == "__main__":
    from src.embed_store import load_index
    from src.retrieve import retrieve

    index, chunks = load_index()
    query = input("Enter a test query: ")
    retrieved = retrieve(query, index, chunks)

    prompt = build_prompt(query, retrieved)
    answer = generate_answer(prompt)

    print("\n--- Answer ---")
    print(answer)

    sentences, citation_map = extract_citations(answer, len(retrieved))
    print("\n--- Parsed citations ---")
    print(citation_map)