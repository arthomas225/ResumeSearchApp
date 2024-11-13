import msal
from office365.sharepoint.client_context import ClientContext
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import re

# Download NLTK data (only need to run once)
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize model for sentence embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize NLTK tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Authenticate and connect to SharePoint
def authenticate_sharepoint(site_url, client_id, client_secret, tenant_id):
    authority_url = f'https://login.microsoftonline.com/{tenant_id}'
    app = msal.ConfidentialClientApplication(client_id=client_id, client_credential=client_secret, authority=authority_url)
    result = app.acquire_token_for_client(scopes=[f'{site_url}/.default'])

    if 'access_token' in result:
        access_token = result['access_token']
        ctx = ClientContext(site_url).with_access_token(access_token)
        print("Successfully authenticated to SharePoint.")
        return ctx
    else:
        raise Exception("Authentication failed.")

# Read and preprocess text from a SharePoint document
def get_document_text(file_url, ctx):
    try:
        response = ctx.web.get_file_by_server_relative_url(file_url).download().execute_query()
        content = response.content.decode('utf-8')
        return preprocess_text(content)
    except Exception as e:
        print(f"Error retrieving file: {e}")
        return None

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    tokens = [word for word in tokens if word not in stop_words]
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmas)

# Cosine similarity search function
def search_resumes(ctx, library_name, query, top_n=5):
    query_embedding = model.encode(preprocess_text(query))
    library = ctx.web.lists.get_by_title(library_name)
    files = library.root_folder.files
    ctx.load(files)
    ctx.execute_query()

    resume_scores = []
    for file in files:
        file_url = file.serverRelativeUrl
        resume_text = get_document_text(file_url, ctx)
        if resume_text:
            resume_embedding = model.encode(resume_text)
            similarity = cosine_similarity([query_embedding], [resume_embedding])[0][0]
            resume_scores.append((file.name, similarity))

    ranked_resumes = sorted(resume_scores, key=lambda x: x[1], reverse=True)[:top_n]
    print("Top matching resumes:")
    for idx, (name, score) in enumerate(ranked_resumes, start=1):
        print(f"{idx}. {name} - Similarity: {score:.2f}")

# Main function to execute the search
if __name__ == '__main__':
    # Replace with your SharePoint credentials
    site_url = 'https://longeneckerassociates.sharepoint.com/sites/ProjectDelivery'
    library_name = 'RESUMES'
    client_id = 'your_client_id'
    client_secret = 'your_client_secret'
    tenant_id = 'your_tenant_id'

    # Authenticate and initialize SharePoint context
    ctx = authenticate_sharepoint(site_url, client_id, client_secret, tenant_id)

    # Define your search query
    query = "Looking for candidates with experience in project management and data analytics."

    # Perform cosine similarity search
    search_resumes(ctx, library_name, query, top_n=5)
