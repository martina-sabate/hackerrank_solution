# Read input from STDIN. Print output to STDOUT

import sys
import math
from collections import Counter

# Sample STDIN:
#   They bought a red apples from the market.
#   An apple pie is my favorite dessert.
#   Oranges and apples are often compared.
#   I prefer bananas over oranges any day.

# Take STDIN and format it
def format_input():
    docs = sys.stdin.read().splitlines() # reads all input lines
    [doc.strip() for doc in docs]
    return docs

# Tokenise text (splits into terms)
def tokenise(text):
    return text.lower().split()

# Compute Term Frequency score (TF)
def compute_tf(doc_tokenised):
    tf = Counter(doc_tokenised)
    total_terms = len(doc_tokenised)
    return {term: count / total_terms for term, count in tf.items()}

# Computes Inverse Document Frequency score (IDF)
def compute_idf(all_docs_tokens):
    N = len(all_docs_tokens)
    idf = {}
    all_terms = set(term for doc in all_docs_tokens for term in doc)
    for term in all_terms:
        containing = sum(term in doc for doc in all_docs_tokens)
        idf[term] = math.log((N + 1) / (containing + 1)) + 1  
        # Calculates smooth IDF. 
        # Adds +1 inside parenthesis to avoid /0 or log(0). 
        # Adds +1 outside parenthesis to shift weights to positive (1.5 instead of 0.5).
        
    return idf

# Compute TF-IDF score
def compute_tfidf(tf, idf):
    return {term: tf.get(term, 0) * idf.get(term, 0) for term in idf}

# Computes Cosine Similarity (between 2 documents)
def cosine_similarity(vec1, vec2):
    common_terms = set(vec1.keys()) | set(vec2.keys())
    dot = sum(vec1.get(term, 0) * vec2.get(term, 0) for term in common_terms)
    norm1 = math.sqrt(sum(val**2 for val in vec1.values()))
    norm2 = math.sqrt(sum(val**2 for val in vec2.values()))
    return dot / (norm1 * norm2) if norm1 and norm2 else 0.0

if __name__ == "__main__":

    # Step 1: Get input (split in docs)
    docs = format_input()
    
    # Step 2: Tokenise each doc (splits in terms)
    tokenised_docs = [tokenise(doc) for doc in docs]
    
    # Step 3: Compute Term Frequency score for each doc
    tfs = [compute_tf(doc) for doc in tokenised_docs]
    
    # Step 4: Compute Inverse Document Frequency across docs
    idf = compute_idf(tokenised_docs)
    
    # Step 5: Compute TF-IDF score for each one of the terms
    tfidf_vectors = [compute_tfidf(tf, idf) for tf in tfs]
    
    # Step 6: Perform similarity search (2,3,4 with respect to 1)
    similarities = [
        cosine_similarity(tfidf_vectors[0], tfidf_vectors[1]),  # 1 vs 2
        cosine_similarity(tfidf_vectors[0], tfidf_vectors[2]),  # 1 vs 3
        cosine_similarity(tfidf_vectors[0], tfidf_vectors[3])   # 1 vs 4
    ]
    
    # Define output labels
    labels = [2, 3, 4]

    # Find most similar
    most_similar = labels[similarities.index(max(similarities))]
    
    print(most_similar) # Returns 2 as expected