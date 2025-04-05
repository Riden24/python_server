import numpy as np
import math
# import a python function from text_processing.py
from text_processing import preprocess_text
from text_processing import build_inverted_index
import json

def compute_tf(freq, doc_length, max_freq = 0, method="augmented"):
    if method == "raw":
        return freq / doc_length
    elif method == "log":
        return 1 + math.log(freq) if freq > 0 else 0
    elif method == "augmented":
        return 0.5 + (0.5 * freq / max_freq)
    return 0

def compute_cosine_similarity(tf_idf, query, inverted_index, N):
    """Compute cosine similarity between query and documents."""
    query_tf_idf = {}

    # Compute TF-IDF for query using the same IDF values as the document matrix
    for term in query:
        if term in inverted_index:
            df = len(inverted_index[term])  # Document frequency
            idf = math.log(N / df) if df > 0 else 0  # Compute IDF
            tf = compute_tf(query.count(term), len(query), method="raw")  # Query term frequency
            query_tf_idf[term] = tf * idf  # TF-IDF for query

    # Compute cosine similarity for each document
    scores = {}
    for doc_id, doc_vector in tf_idf.items():
        doc_norm = np.linalg.norm(list(doc_vector.values()))  # Document vector norm
        query_norm = np.linalg.norm(list(query_tf_idf.values()))  # Query vector norm

        # Compute dot product
        dot_product = sum(doc_vector.get(term, 0) * query_tf_idf.get(term, 0) for term in query_tf_idf)

        # Compute cosine similarity
        if doc_norm > 0 and query_norm > 0:
            scores[doc_id] = dot_product / (doc_norm * query_norm)
        else:
            scores[doc_id] = 0

    # Normalize scores
    # if type(scores) == str:
    #     print(scores)
    # scores = min_max_normalize(scores)
    
    # Rank documents by score
    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:100]
    return ranked_docs

def display_results(ranked_results, doc_lookup, top_n=10):
    """Display the top N ranked image search results with their titles and allow the user to view one."""
    print("\n🔍 Top Image Search Results:\n")
    for rank, (doc_id, score) in enumerate(ranked_results[:top_n], start=1):
        title = doc_lookup.get(doc_id, ("Unknown Title", ""))[0]
        print(f"{rank}. 📄 [{doc_id}] — {title} (Score: {score:.4f})")

    doc_id_to_read = input("\nEnter an image ID to view details (e.g., image_2.jpg), or press Enter to skip: ").strip()

    if doc_id_to_read in doc_lookup:
        title, description = doc_lookup[doc_id_to_read]
        print(f"\n📘 === {title} ===")
        print(description[:500] + ("..." if len(description) > 500 else ""))
    elif doc_id_to_read == "":
        print("✅ No document selected.")
    else:
        print("❌ Invalid image ID.")

# query_text = "red frog"
# query_text = preprocess_text(query_text)  # Apply the same preprocessing as indexing
# print(query_text)
# # read json file tf_idf_matrix.json
# with open("tf_idf_matrix.json", "r") as json_file:
#     tf_idf = json.load(json_file)

# with open("inverted_index_matrix.json", "r") as json_file:
#     inverted_index = json.load(json_file)

# with open("fused_image_data.json", "r") as json_file:
#     documents = json.load(json_file)


# ranked_docs = compute_cosine_similarity(tf_idf, query_text, inverted_index, len(documents))

# doc_lookup = {
#     f"{i}": (
#         doc.get("title", "Unknown Title"),
#         doc.get("ImageDescription", "") or doc.get("Artist", "") or doc.get("title", "")
#     )
#     for i, doc in enumerate(documents)
# }




# display_results(ranked_docs, doc_lookup)
# print(ranked_docs)