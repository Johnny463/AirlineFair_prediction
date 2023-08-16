import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# CORS settings
origins = ["*"]  # Adjust this to limit origins if needed
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])

# Load the small model and tokenizer
model_small = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer_small = T5Tokenizer.from_pretrained('t5-small')
device = torch.device('cpu')

class SummaryRequest(BaseModel):
    text: str

def get_summary(tokenizer, model, prepared_text):
    tokenized_text = tokenizer.encode("summarize: " + prepared_text, return_tensors="pt").to(device)
    summary_ids = model.generate(tokenized_text,
                                 num_beams=4,
                                 no_repeat_ngram_size=2,
                                 min_length=30,
                                 max_length=100,
                                 early_stopping=True)
    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return output

@app.get("/")
def home_endpoint():
    return 'Nothing to see here... move along.'

@app.post("/summary")
@app.post("/summary")
async def get_summary_endpoint(text: str):
    t5_prepared_text = text.strip().replace("\n", "")
    if not t5_prepared_text:
        raise HTTPException(status_code=400, detail="No input text provided")

    output = get_summary(tokenizer_small, model_small, t5_prepared_text)
    return {"summary": output}
