"""In-memory vector retrieval using OpenAI embeddings."""

import os
from typing import List, Dict, Optional
import numpy as np
from openai import OpenAI


def get_openai_client(api_key: str = None) -> OpenAI:
    """Get OpenAI client instance."""
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

    return OpenAI(api_key=api_key)


def embed_text(text: str, api_key: str = None) -> np.ndarray:
    """
    Get embedding vector for a single text string using OpenAI.

    Args:
        text: Text to embed
        api_key: OpenAI API key (optional, will use env var if not provided)

    Returns:
        Numpy array of embedding vector
    """
    client = get_openai_client(api_key)

    # Use text-embedding-3-small (cost-effective and good quality)
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
        encoding_format="float"
    )

    # Extract embedding vector
    embedding = response.data[0].embedding

    return np.array(embedding)


def embed_all_chunks(chunks: List[Dict], api_key: str = None, batch_size: int = 100) -> np.ndarray:
    """
    Embed all chunks in batches for efficiency.

    Args:
        chunks: List of chunk dictionaries with "text" field
        api_key: OpenAI API key (optional)
        batch_size: Number of chunks to embed per API call

    Returns:
        Numpy array of shape (n_chunks, embedding_dim)
    """
    client = get_openai_client(api_key)

    all_embeddings = []

    # Process in batches
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [chunk["text"] for chunk in batch]

        # Get embeddings for batch
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
            encoding_format="float"
        )

        # Extract embeddings
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return np.array(all_embeddings)


def cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between query and document vectors.

    Args:
        query_vec: Query embedding vector (1D array)
        doc_vecs: Document embedding vectors (2D array, shape: [n_docs, embedding_dim])

    Returns:
        1D array of similarity scores
    """
    # Normalize vectors
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    doc_norms = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-10)

    # Compute dot product (cosine similarity for normalized vectors)
    similarities = np.dot(doc_norms, query_norm)

    return similarities


def retrieve_top_k(
    query: str,
    chunks: List[Dict],
    vectors: np.ndarray,
    k: int = 2,
    page_filter: Optional[List[int]] = None,
    api_key: str = None
) -> List[Dict]:
    """
    Retrieve top-k most relevant chunks for a query using cosine similarity.

    Args:
        query: User's question
        chunks: List of chunk dictionaries
        vectors: Embedding vectors for all chunks (2D numpy array)
        k: Number of chunks to retrieve (default: 2)
        page_filter: Optional list of page numbers to filter by
        api_key: OpenAI API key (optional)

    Returns:
        List of top-k chunk dictionaries with added "score" field
    """
    if len(chunks) == 0 or len(vectors) == 0:
        return []

    # Embed the query
    query_vec = embed_text(query, api_key)

    # Apply page filter if provided
    if page_filter:
        filtered_indices = [i for i, chunk in enumerate(chunks) if chunk["page"] in page_filter]
        if not filtered_indices:
            # No chunks match the filter
            return []

        filtered_chunks = [chunks[i] for i in filtered_indices]
        filtered_vectors = vectors[filtered_indices]
    else:
        filtered_chunks = chunks
        filtered_vectors = vectors
        filtered_indices = list(range(len(chunks)))

    # Compute similarities
    similarities = cosine_similarity(query_vec, filtered_vectors)

    # Get top-k indices
    top_k_local_indices = np.argsort(similarities)[-k:][::-1]  # Descending order

    # Get top-k chunks with scores
    results = []
    for local_idx in top_k_local_indices:
        chunk = filtered_chunks[local_idx].copy()
        chunk["score"] = float(similarities[local_idx])
        results.append(chunk)

    return results


def retrieve_with_reranking(
    query: str,
    chunks: List[Dict],
    vectors: np.ndarray,
    k: int = 2,
    initial_k: int = 10,
    api_key: str = None
) -> List[Dict]:
    """
    Two-stage retrieval: first retrieve more candidates, then rerank.

    This is a simple version that just retrieves initial_k candidates
    and returns top k. For production, you could add actual reranking
    using a cross-encoder model.

    Args:
        query: User's question
        chunks: List of chunk dictionaries
        vectors: Embedding vectors for all chunks
        k: Final number of chunks to return
        initial_k: Number of candidates to retrieve before reranking
        api_key: OpenAI API key (optional)

    Returns:
        List of top-k reranked chunks
    """
    # For now, just retrieve more and return top-k
    # In future, could add reranking logic here
    candidates = retrieve_top_k(query, chunks, vectors, k=min(initial_k, len(chunks)), api_key=api_key)
    return candidates[:k]


def get_context_window(
    chunk: Dict,
    chunks: List[Dict],
    window_size: int = 1
) -> str:
    """
    Get surrounding context for a chunk (previous and next chunks from same page).

    Args:
        chunk: Target chunk dictionary
        chunks: All chunks
        window_size: Number of chunks before/after to include

    Returns:
        Combined text with context
    """
    page = chunk["page"]
    chunk_id = chunk["chunk_id"]

    # Get chunks from same page
    page_chunks = [c for c in chunks if c["page"] == page]
    page_chunks.sort(key=lambda x: x["chunk_id"])

    # Find current chunk index in page chunks
    try:
        current_idx = next(i for i, c in enumerate(page_chunks) if c["chunk_id"] == chunk_id)
    except StopIteration:
        # Chunk not found, return original
        return chunk["text"]

    # Get window
    start_idx = max(0, current_idx - window_size)
    end_idx = min(len(page_chunks), current_idx + window_size + 1)

    # Combine texts
    context_texts = [page_chunks[i]["text"] for i in range(start_idx, end_idx)]
    return "\n\n".join(context_texts)


def format_retrieved_context(retrieved_chunks: List[Dict], include_scores: bool = False) -> str:
    """
    Format retrieved chunks into a readable context string for LLM.

    Args:
        retrieved_chunks: List of chunk dictionaries with scores
        include_scores: Whether to include similarity scores in output

    Returns:
        Formatted context string
    """
    if not retrieved_chunks:
        return "No relevant context found."

    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        score_info = f" (relevance: {chunk['score']:.2f})" if include_scores and "score" in chunk else ""
        context_parts.append(
            f"[Source {i} - Page {chunk['page']}{score_info}]\n{chunk['text']}"
        )

    return "\n\n---\n\n".join(context_parts)


def get_unique_pages(chunks: List[Dict]) -> List[int]:
    """
    Extract unique page numbers from retrieved chunks.

    Args:
        chunks: List of chunk dictionaries

    Returns:
        Sorted list of unique page numbers
    """
    pages = set(chunk["page"] for chunk in chunks)
    return sorted(list(pages))
