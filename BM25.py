import math
from collections import defaultdict
from text_processing import preprocess_text
import json
from compute_cosine import display_results

def compute_bm25(inverted_index, documents, query, k1=1.5, b=0.75):
    """
    Compute BM25 scores for documents given a single query.
    
    Args:
        inverted_index (dict): Inverted index mapping terms to (doc_id, tf) pairs.
        documents (list): List of documents, where each document is a tuple (doc_id, title, author, bib, text).
        query (str): The query text.
        k1 (float): Controls term frequency saturation. Default is 1.5.
        b (float): Controls document length normalization. Default is 0.75.
    
    Returns:
        list: Sorted BM25 scores for the query, as a list of tuples (doc_id, score).
    """
    N = len(documents)  # Total number of documents
    doc_lengths = {
        doc.get("title", ""): len(preprocess_text(
            doc.get("ImageDescription", "") or doc.get("Artist", "") or doc.get("title", "") or doc.get("image_url", "") or doc.get("file_page_url", "") or doc.get("caption", "") 
        ))
        for doc in documents
    }
    avg_doc_length = sum(doc_lengths.values()) / N  # Average document length
    
    # Compute IDF for each term using Robertson-Sparck Jones formula
    idf = {}
    for term, doc_freqs in inverted_index.items():
        df = len(doc_freqs)  # Number of documents containing the term
        idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1)
    
    bm25_scores = defaultdict(float)  # Store BM25 scores, default to 0
    
    query_terms = preprocess_text(query)  # Preprocess the query text
    
    for term in query_terms:
        if term in inverted_index:
            for doc_id, tf in inverted_index[term]:
                doc_length = doc_lengths.get(doc_id, avg_doc_length)
                    
                # Compute TF weight with saturation and document length normalization
                tf_weight = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_length / avg_doc_length)))
                    
                # Add BM25 score contribution for this term
                bm25_scores[doc_id] += idf[term] * tf_weight
    
    # Convert BM25 scores to a sorted list of tuples (doc_id, score) sorted by score in descending order
    sorted_bm25_scores = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_bm25_scores



with open("tf_idf_matrix.json", "r") as f:
    tf_idf = json.load(f)
with open("inverted_index_matrix.json", "r") as f:
    inverted_index = json.load(f)
with open("fused_image_data.json", "r") as f:
    documents = json.load(f)


doc_lookup = {
    f"{i}": (
        doc.get("title", "Unknown Title"),
        doc.get("ImageDescription", "") or doc.get("Artist", "") or doc.get("title", "")
    )
    for i, doc in enumerate(documents)
}

# custom_query = "red frog"
# query_dict = {"custom_query": custom_query}  # Format query as a dictionary

# bm25_scores = compute_bm25(inverted_index, documents, custom_query)
# print(bm25_scores)

# # Ranking
# # ranked_docs = sorted(bm25_scores["custom_query"].items(), key=lambda x: x[1], reverse=True)[:10]

# display_results(bm25_scores, doc_lookup, top_n=10)