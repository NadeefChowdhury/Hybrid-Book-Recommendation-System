import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# =========================
# LOAD DATA
df = pd.read_csv(filename.csv)
df = df[~df['Genres'].str.lower().str.strip().str.contains('other', na=False)]
# =========================
print("------------DATASET INFO--------------")
print(df.info())
print("--------------------------------------")

# =========================
# BASIC CLEANING
# =========================
df['Book-Title'] = df['Book-Title'].str.lower()

df['Review'] = df['Review'].fillna('')
df['Genres'] = df['Genres'].fillna('')
df['Book-Author'] = df['Book-Author'].fillna('')
df['Publisher'] = df['Publisher'].fillna('')
df['Year'] = df['Year-Of-Publication'].astype(str)

# Remove duplicate USER-BOOK interactions ONLY
df = df.drop_duplicates(subset=['User-ID', 'ISBN'])

# =========================
# COLLABORATIVE FILTERING
# =========================
df_cf = df.copy()

pivot = df_cf.pivot_table(
    index='User-ID',
    columns='ISBN',
    values='Book-Rating'
).fillna(0)

# Mean-center (CRUCIAL)
pivot = pivot.subtract(pivot.mean(axis=1), axis=0).fillna(0)

cf_model = NearestNeighbors(metric='cosine', algorithm='brute')
cf_model.fit(pivot.T)

isbn_list = list(pivot.columns)

def get_cf_neighbors(isbn, n=10):
    if isbn not in isbn_list:
        return {}
    
    idx = isbn_list.index(isbn)
    
    distances, indices = cf_model.kneighbors(
        pivot.T.iloc[idx].values.reshape(1, -1),
        n_neighbors=n
    )
    
    results = {}
    for i in range(1, len(indices[0])):
        results[isbn_list[indices[0][i]]] = distances[0][i]
    
    return results

# =========================
# CONTENT-BASED
# =========================
# ONE ROW PER BOOK
df_content = df.drop_duplicates(subset='ISBN').reset_index(drop=True)
rating_stats = df.groupby('ISBN').agg({
    'Book-Rating': ['mean', 'count']
}).reset_index()

rating_stats.columns = ['ISBN', 'avg_rating', 'rating_count']
df_content = df_content.merge(rating_stats, on='ISBN', how='left')

df_content['avg_rating'] = df_content['avg_rating'].fillna(0)
df_content['rating_count'] = df_content['rating_count'].fillna(0)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df_content[['avg_rating', 'rating_count']] = scaler.fit_transform(
    df_content[['avg_rating', 'rating_count']]
)
from scipy.sparse import csr_matrix

mat_rating = csr_matrix(df_content[['avg_rating']].values)
mat_count = csr_matrix(df_content[['rating_count']].values)
tfidf = TfidfVectorizer(stop_words='english')

# Vectorize each column separately
mat_genre = tfidf.fit_transform(df_content['Genres'])
mat_author = tfidf.fit_transform(df_content['Book-Author'])
mat_publisher = tfidf.fit_transform(df_content['Publisher'])
mat_year = tfidf.fit_transform(df_content['Year'])
mat_review = tfidf.fit_transform(df_content['Review'])

# APPLY WEIGHTS (you can tune these)
from scipy.sparse import hstack

content_matrix = hstack([
    mat_genre * 0.30,
    mat_author * 0.25,
    mat_publisher * 0.05,
    mat_review * 0.05,
    mat_rating * 0.15,
    mat_count * 0.2
])
# Normalize AFTER combining
content_matrix = normalize(content_matrix)

content_model = NearestNeighbors(metric='cosine', algorithm='brute')
content_model.fit(content_matrix)

def get_content_neighbors(isbn, n=10):
    if isbn not in df_content['ISBN'].values:
        return {}
    
    idx = df_content[df_content['ISBN'] == isbn].index[0]
    
    distances, indices = content_model.kneighbors(
        content_matrix[idx],
        n_neighbors=n
    )
    
    results = {}
    for i in range(1, len(indices[0])):
        results[df_content.iloc[indices[0][i]]['ISBN']] = distances[0][i]
    
    return results

# =========================
# NORMALIZATION
# =========================
def minmax(scores):
    if not scores:
        return []
    mn, mx = min(scores), max(scores)
    if mx - mn == 0:
        return [0]*len(scores)
    return [(s - mn) / (mx - mn) for s in scores]

# =========================
# TITLE → ISBN
# =========================
def get_isbn_from_title(title):
    matches = df[df['Book-Title'].str.contains(title.lower(), na=False)]
    if matches.empty:
        return None
    return matches.iloc[0]['ISBN']

# =========================
# HYBRID RECOMMENDER
# =========================
def recommend_books(title, n=5, alpha=0.5):
    
    isbn = get_isbn_from_title(title)
    
    if isbn is None:
        return "Book not found"
    
    cf = get_cf_neighbors(isbn, n*3)
    content = get_content_neighbors(isbn, n*3)
    
    all_books = list(set(cf.keys()) | set(content.keys()))
    
    # Convert distance → similarity
    cf_scores = [1 - cf.get(b, 1) for b in all_books]
    content_scores = [1 - content.get(b, 1) for b in all_books]
    
    # Normalize
    cf_norm = minmax(cf_scores)
    content_norm = minmax(content_scores)
    
    # Combine
    final_scores = []
    for i, b in enumerate(all_books):
        score = alpha * cf_norm[i] + (1 - alpha) * content_norm[i]
        final_scores.append((b, score))
    
    # Sort
    final_scores = sorted(final_scores, key=lambda x: x[1], reverse=True)
    
    # Return results
    results = []
    for isbn, score in final_scores[:n]:
        book = df_content[df_content['ISBN'] == isbn].iloc[0]
        results.append({
            'Title': book['Book-Title'],
            'Author': book['Book-Author'],
            'Genres': book['Genres'],
            'Score': score
        })
    
    for result in results:
        print(result)

# =========================
# TEST
# =========================
print(recommend_books("Harry Potter", n=10, alpha=0.5))
