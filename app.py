import pickle
import re
import nltk
from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

# تحميل الموارد الضرورية
nltk.download(['punkt', 'wordnet', 'stopwords'], quiet=True)

app = FastAPI()

# تأكد من إنشاء مجلد اسمه static ومجلد اسمه templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# تحميل ملفات الموديل
try:
    with open('count_vectorizer.pkl', 'rb') as f: cv = pickle.load(f)
    with open('tfidf_transformer.pkl', 'rb') as f: tfidf_transformer = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f: feature_names = pickle.load(f)
except FileNotFoundError:
    print("Error: Pickle files not found. Make sure they are in the same directory.")

def preprocess_text(txt):
    txt = txt.lower()
    txt = re.sub(r"<.*?>", " ", txt)
    txt = re.sub(r"[^a-zA-Z]", " ", txt)
    txt = nltk.word_tokenize(txt)
    stop_words = set(stopwords.words("english"))
    txt = [word for word in txt if word not in stop_words and len(word) >= 3]
    lmtr = WordNetLemmatizer()
    txt = [lmtr.lemmatize(word) for word in txt]
    return " ".join(txt)

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    sorted_items = sorted_items[:topn]
    results = {feature_names[idx]: round(score, 3) for idx, score in sorted_items}
    return results

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/extract_keywords", response_class=HTMLResponse)
async def extract_keywords(request: Request, file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode('utf-8', errors='ignore')
    preprocessed_text = preprocess_text(text)
    tf_idf_vector = tfidf_transformer.transform(cv.transform([preprocessed_text]))
    sorted_items = sort_coo(tf_idf_vector.tocoo())
    keywords = extract_topn_from_vector(feature_names, sorted_items, 20)
    
    # التعديل هنا: نرسل النتائج لنفس الصفحة الرئيسية
    return templates.TemplateResponse("index.html", {"request": request, "keywords": keywords})

@app.post("/search_keywords", response_class=HTMLResponse)
async def search_keywords(request: Request, search: str = Form(...)):
    results = [kw for kw in feature_names if search.lower() in kw.lower()][:20]
    
    
    return templates.TemplateResponse("index.html", {"request": request, "search_results": results, "query": search})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)