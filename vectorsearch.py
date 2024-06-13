import numpy as np
import faiss
import pandas as pd
import spacy

# Load spaCy model with pre-trained word embeddings
nlp = spacy.load("en_core_web_md")

# Sample news articles for each category
politics_articles = [
    "Political unrest continues in the region as protests escalate.",
    "Government announces new policies to address economic challenges.",
    "Opposition leaders call for nationwide strike against recent legislation.",
    "President delivers State of the Union address outlining key priorities.",
    "Diplomatic tensions rise between neighboring countries over border disputes.",
    "Political parties gear up for upcoming elections with intense campaigning.",
    "Lawmakers debate controversial bill on healthcare reform in parliament.",
    "Corruption scandal rocks government as officials face investigation.",
    "Supreme Court issues landmark ruling on civil rights case.",
    "Citizens express concerns over rising unemployment rates and inflation."
]

sports_articles = [
    "Team clinches victory in championship match with a last-minute goal.",
    "Star athlete breaks world record in sprint event at international competition.",
    "Major league baseball playoffs heat up as teams vie for championship title.",
    "Basketball superstar leads team to victory with stellar performance.",
    "Olympic gold medalist announces retirement from professional sports.",
    "Football club signs lucrative sponsorship deal with multinational corporation.",
    "Tennis prodigy emerges as rising star with impressive performance at Grand Slam.",
    "Golf tournament attracts top players from around the world to compete.",
    "Athletics championship showcases exceptional talent in track and field events.",
    "Cricket team celebrates historic win in test match against arch-rival."
]

technology_articles = [
    "Tech giant unveils latest smartphone model with innovative features.",
    "Artificial intelligence startup secures funding for groundbreaking research.",
    "New software update promises enhanced security and performance for users.",
    "Virtual reality headset offers immersive gaming experience for enthusiasts.",
    "Blockchain technology revolutionizes finance industry with decentralized system.",
    "Social media platform introduces new algorithm to prioritize user content.",
    "Cybersecurity firm warns of increasing cyber threats targeting businesses.",
    "Robotics company develops humanoid robot capable of human-like interactions.",
    "E-commerce platform launches new service for personalized shopping recommendations.",
    "Internet of Things devices connect homes and businesses for greater convenience."
]

# Create DataFrame
politics_df = pd.DataFrame({'Text': politics_articles, 'Label': ['politics'] * 10})
sports_df = pd.DataFrame({'Text': sports_articles, 'Label': ['sports'] * 10})
technology_df = pd.DataFrame({'Text': technology_articles, 'Label': ['technology'] * 10})

# Concatenate DataFrames
news_df = pd.concat([politics_df, sports_df, technology_df], ignore_index=True)

# Convert text data into semantic embeddings
data_vectors = np.array([nlp(text).vector for text in news_df['Text']], dtype='float32')

# Indexing
def build_index(data):
    d = data.shape[1]  # Dimensionality of vectors
    index = faiss.IndexFlatL2(d)  # Construct the index
    index.add(data)  # Add data vectors to the index
    return index

index = build_index(data_vectors)

# Querying
def search(index, query_text, k=5):
    query_vector = nlp(query_text).vector.reshape(1, -1).astype('float32')
    D, I = index.search(query_vector, k)  # Search the index
    return D, I

# Sample query
query_text = "New policies announced to tackle economic challenges"

# Perform search
distances, indices = search(index, query_text)

# Print results
from time import time
st = time()
print("Query Text:", query_text)
print("Indices of Nearest Neighbors:", indices)
print("Corresponding Labels:", news_df['Label'].iloc[indices.ravel()].tolist())
print("Time Elapsed : {} milliseconds".format(round((time()-st)*1000,4)))
