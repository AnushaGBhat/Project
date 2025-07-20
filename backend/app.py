from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

MODEL_PATH = "./model"
if os.path.exists(MODEL_PATH):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
else:
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

class TextIn(BaseModel):
    text: str

@app.post("/predict")
def predict(input: TextIn):
    result = classifier(input.text)[0]
    return {
        "label": result["label"].lower(),
        "score": round(result["score"], 4)
    }
