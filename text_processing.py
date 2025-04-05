
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')  # Download WordNet data
nltk.download('stopwords')  # Download stopwords list
from nltk.tokenize import word_tokenize
import json

# "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/35/Strawberry_poison_dart_frog_%2870539%29.jpg/500px-Strawberry_poison_dart_frog_%2870539%29.jpg",
# "file_page_url": "https://en.wikipedia.org/wiki/File:Strawberry_poison_dart_frog_(70539).jpg",

import json

import json

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        # Load the entire JSON array at once since it's not a JSONL file
        data = json.load(f)
    return data





def preprocess_text(text):
    # Remove newline characters and extra spaces
    
    prefix = "https://"
    if text.startswith(prefix):
        # Remove the prefix
        text = text[len(prefix):]
        text = re.sub(r'\w*\d+\w*', '', text)
    
    # Remove special characters and keep only alphanumeric characters and underscores
    text = re.sub(r'[^a-zA-Z0-9_]', ' ', text)

    # Replace underscores with space (optional based on your needs)
    text = text.replace('_', ' ')
    text = re.sub(r'\s+', ' ', text)  # Replace all kinds of whitespace with a single space
    text = text.strip()  # Remove leading/trailing whitespace
    text = text.lower()

    text = re.sub(r'[^\w\s]', '', text)  # Keep only alphanumeric and whitespace characters
    # remove words that contain a number and then px
    
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    # add words manually to stop_words
    stop_words.update(["https", "thumb", "upload", "wikimedia", "org", "commons", "wikipedia", "jpg", "png", "xml", "jpeg", "gif", "svg", "en", "wiki", "file", ])
    # print(stop_words)
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Apply stemming
    stemmer = PorterStemmer()
    filtered_tokens = [stemmer.stem(word) for word in filtered_tokens]
    return filtered_tokens


from collections import defaultdict

def build_inverted_index(documents):
    inverted_index = defaultdict(list)  # {term: [(doc_id, freq), ...]}
    doc_id = 0
    for document in documents:
        title = document.get("title", "")  
        image_url = document.get("image_url", "") 
        file_page_url = document.get("file_page_url", "") 
        artist = document.get("Artist", "")  
        image_description = document.get("ImageDescription", "")  
        caption = document.get("caption", "")
        # clip_label = document.get("clip_label", "")


        
        if not title:  # Skip documents without text
            continue
        else:
            processed_text = preprocess_text(title + " " + image_url + " " + file_page_url + " " + artist + " " + image_description + " " + caption + "")  # Combine title & text
            term_freq = defaultdict(int)
            
            # Count term frequency in the document
            for term in processed_text:
                term_freq[term] += 1
            
            # Add term and frequency to the inverted index
            for term, freq in term_freq.items():
                inverted_index[term].append((doc_id, freq))
        
        doc_id += 1
    return inverted_index

import math
from collections import defaultdict

import math
from collections import defaultdict

def compute_tf_idf(inverted_index, documents):
    """Compute the TF-IDF matrix from an inverted index."""
    N = len(documents)  # Total number of documents
    tf_idf = defaultdict(dict)  # Store TF-IDF scores

    # Map document titles to their preprocessed text lengths
    doc_lengths = {
        doc.get("title", ""): len(preprocess_text(
            doc.get("ImageDescription", "") or doc.get("Artist", "") or doc.get("title", "") or doc.get("image_url", "") or doc.get("file_page_url", "") or doc.get("caption", "") 
        ))
        for doc in documents
    }

    avg_doc_length = sum(doc_lengths.values()) / N if N > 0 else 1  # Prevent division by zero

    for term, doc_freqs in inverted_index.items():
        df = len(doc_freqs)  # Number of documents containing the term

        if df == 0:
            print(f"Warning: Term '{term}' has df=0, check indexing!")

        idf = math.log((N + 1) / (df + 1)) + 1  # Smoothed IDF to avoid zero division

        for doc_id, freq in doc_freqs:
            doc_length = doc_lengths.get(doc_id, avg_doc_length)  # Fallback to avg if not found
            tf = (1 + math.log(freq)) if freq > 0 else 0

            # BM25-like normalization
            tf /= (1 - 0.75 + 0.75 * (doc_length / avg_doc_length))

            tf_idf_score = tf * idf
            tf_idf[doc_id][term] = tf_idf_score

    return tf_idf




# # test inverting matrix
# documents = read_json('./fused_image_data.json')
# inverted_index = build_inverted_index(documents)
# print(dict(list(inverted_index.items())[:5]))  # Display first 5 terms in the index
# with open("inverted_index_matrix.json", "w") as json_file:
#     json.dump(inverted_index, json_file, indent=4)

# tf_idf = compute_tf_idf(inverted_index, documents)
# print(dict(list(tf_idf.items())[:5]))  # Display first 5 documents in the TF-IDF matrix

# # save tf_idf matrix to json
# with open("tf_idf_matrix.json", "w") as json_file:
#     json.dump(tf_idf, json_file, indent=4)


# # test the preprocessing
# example_text = "The quick brown fox is running fast but it looks like he is flying for fuck sake! It got 6 eyes for some reason... "
# processed_text = preprocess_text(example_text)
# print(processed_text)

# example_text = "https://en.wikipedia.org/wiki/File:Alfred_Waud_by_Timothy_H._O%27Sullivan.jpg"
# processed_text = preprocess_text(example_text)
# print(processed_text)


