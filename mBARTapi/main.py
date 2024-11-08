from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load MBart model and tokenizer
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

# Helper function to highlight errors
def highlight_errors(original, corrected):
    original_tokens = original.split()
    corrected_tokens = corrected.split()
    highlighted = []
    for o_token, c_token in zip(original_tokens, corrected_tokens):
        if o_token != c_token:
            highlighted.append(f"<del style='color:red;'>{o_token}</del> <span style='color:green;'>{c_token}</span>")
        else:
            highlighted.append(o_token)
    return " ".join(highlighted)

# Grammar correction with mBART
def correct_grammar_mbart(text, src_lang):
    tokenizer.src_lang = src_lang
    encoded_input = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    generated_tokens = model.generate(
        **encoded_input,
        max_length=512,
        num_beams=5,
        early_stopping=True,
        forced_bos_token_id=tokenizer.lang_code_to_id[src_lang]
    )
    corrected_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return corrected_text

# API usage: track incorrect words
def get_corrections(original, corrected):
    original_tokens = original.split()
    corrected_tokens = corrected.split()
    corrections = {}
    for o_token, c_token in zip(original_tokens, corrected_tokens):
        if o_token != c_token:
            corrections[o_token] = c_token
    return corrected_tokens, corrections

# Route for the homepage
@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/", response_class=HTMLResponse)
async def post_home(request: Request, text: str = Form(...), lang_code: str = Form(...)):
    corrected_text = correct_grammar_mbart(text, lang_code)
    highlighted_text = highlight_errors(text, corrected_text)
    return templates.TemplateResponse(
        "index.html", {"request": request, "result": highlighted_text, "text": text}
    )

# Data model for API requests
class CorrectionRequest(BaseModel):
    text: str
    lang_code: str

class CorrectionResponse(BaseModel):
    corrected_text: str
    corrections: dict[str, str]  # Key: Incorrect word, Value: Correct word

# API endpoint for external applications
@app.post("/api", response_model=CorrectionResponse)
async def correct_text(request: CorrectionRequest):
    corrected_text = correct_grammar_mbart(request.text, request.lang_code)
    _, corrections = get_corrections(request.text, corrected_text)
    return CorrectionResponse(corrected_text=corrected_text, corrections=corrections)
