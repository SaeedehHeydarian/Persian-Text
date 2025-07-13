#%% md
# ## Importing Libraries
#%%
from nest_asyncio import apply
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from hazm import Normalizer, word_tokenize, stopwords_list
import  pandas as pd
import re
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from sqlalchemy import create_engine, text
from sklearn.decomposition import LatentDirichletAllocation
import seaborn as sns
import matplotlib.pyplot as plt
#%% md
# ## Getting Data from Database 
#%%
user = 'user'
password = 'password'
host = 'host'
port = 'port'
service_name ='srvice_name'

dsn = f'oracle+oracledb://{user}:{password}@{host}:{port}/?service_name={service_name}'
engine = create_engine(dsn)

df = f"""
    Query
"""
with engine.connect() as conn:
    df = pd.read_sql(sql=df, con=conn)

#%%
df=pd.read_csv(r"C:\Users\s.heydarian\Desktop\dashboard data\TRANSACTION_DESCRIPTION_202507071458.csv")
#%% md
# ## Data cleaning
#%%
df.drop_duplicates(inplace=True)
#%%
def remove_all_numbers(text):
    return re.sub(r'[0-9۰-۹]', '', text)
#%%
df = df[~df["NAME"].str.isdigit()]
#%%
def remove_all_english(text):
    return re.sub(r'[a-zA-z]', ''  , text)
#%%
def remove_emojis_and_symbols(text):
    emoji_symbol_pattern = re.compile("["
                                      u"\U0001F600-\U0001F64F"  # Emoticons
                                      u"\U0001F300-\U0001F5FF"  # Symbols & pictographs
                                      u"\U0001F680-\U0001F6FF"  # Transport & map symbols
                                      u"\U0001F700-\U0001F77F"  # Alphanumeric & geometric shapes
                                      u"\U0001F780-\U0001F7FF"  # Geometric shapes extended
                                      u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                                      u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                                      u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                                      u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                                      u"\U00002702-\U000027B0"  # Dingbats
                                      u"\U000024C2-\U0001F251" 
                                      r"\W"  # Non-word characters (symbols)
                                      "]+", flags=re.UNICODE)

    clean_text = re.sub(emoji_symbol_pattern, ' ', text)

    return clean_text
#%%
df['name']=df['name'].apply(remove_emojis_and_symbols)
#%%
df['name']=df['name'].apply(remove_all_english)
#%%
df['name']=df['name'].apply(remove_all_numbers)

#%%
df['name'] = (
    df['name']
    .str.replace(r'["\r\n]', '', regex=True)      # Remove quotes, carriage return, newlines
    .str.replace(r',,', '', regex=True)           # Remove double commas
    .str.replace(r'\s+', ' ', regex=True)         # Collapse multiple spaces
    .str.strip()                                  # Remove leading/trailing whitespace
)
#%%
df = df[df['DESCRIPTION'].str.strip() != '']
#%% md
# ## Preprocessing
#%%
# Persian stopwords

custom_stopwords  = {"کد", "جهت","برداشت", "کارت", "مبلغ", "توسط" ,  "شماره",  "بمبلغ", "رهگيري" ,  "بانک", "مهرايران", "ما" ,  "جغطائی", "الحسنه" , "مدرن", "الحسنه", "رهگیری" , "الحسنه", "قرض"}

all_stopwords = set(stopwords_list()).union(custom_stopwords)

normalizer = Normalizer()

def preprocess(text):
    text = normalizer.normalize(text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in all_stopwords and len(t) > 1]
    return ' '.join(tokens)  # For TF-IDF, we need a string

#%%
df['name'] = df['name'].apply(preprocess)
#%%
tfidf_vectorizer = TfidfVectorizer(min_df=10, max_df=0.75)
doc_term_matrix = tfidf_vectorizer.fit_transform(df['name'])

print(f"TF-IDF Matrix: {doc_term_matrix.shape}")

#%%
num_topics = 5
lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
doc_topic_matrix = lda.fit_transform(doc_term_matrix)

topic_names = [f"Topic {i+1}" for i in range(num_topics)]
doc_topic_df = pd.DataFrame(doc_topic_matrix, columns=topic_names)
print(doc_topic_df.head())
#%%
num_words = 10
feature_names = tfidf_vectorizer.get_feature_names_out()
#%%
for topic_idx, topic in enumerate(lda.components_):
    print(f"\n Topic {topic_idx + 1}")
    top_indices = topic.argsort()[::-1][:num_words]
    for i in top_indices:
        print(f"   {feature_names[i]} ({topic[i]:.3f})")
#%%
# ========== Step 7: Perplexity vs. Topic Count ==========
perplexities = []
topic_counts = range(2, 11)

for k in topic_counts:
    lda_k = LatentDirichletAllocation(n_components=k, random_state=42)
    lda_k.fit(doc_term_matrix)
    perp = lda_k.perplexity(doc_term_matrix)
    perplexities.append(perp)

# ========== Step 8: Plot Perplexity ==========
plt.figure(figsize=(8, 5))
sns.lineplot(x=topic_counts, y=perplexities, marker='o')
plt.title('Perplexity by Topic Count')
plt.xlabel('Number of Topics')
plt.ylabel('Perplexity')
plt.grid(True)
plt.tight_layout()
plt.show()