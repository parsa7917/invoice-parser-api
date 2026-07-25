import os
import json
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Invoice Data Extractor API",
    description="Extracts structured JSON data from raw invoice text using free Groq LLM.",
    version="1.0.0"
)

# Initialize Client using Groq's Free OpenAI-compatible API
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

class InvoiceItem(BaseModel):
    description: str = Field(description="Description of the item or service")
    quantity: Optional[float] = Field(default=1.0, description="Quantity of items")
    unit_price: Optional[float] = Field(default=0.0, description="Price per unit")

class InvoiceExtractionResponse(BaseModel):
    vendor_name: Optional[str] = Field(default=None, description="Name of the seller or merchant")
    date: Optional[str] = Field(default=None, description="Transaction date in YYYY-MM-DD format")
    total_amount: Optional[float] = Field(default=0.0, description="Total amount paid")
    currency: Optional[str] = Field(default="USD", description="Currency symbol or 3-letter code")
    items: List[InvoiceItem] = Field(default_factory=list, description="List of items purchased")
    tax_amount: Optional[float] = Field(default=0.0, description="Total tax amount if applicable")

class InvoiceExtractionRequest(BaseModel):
    raw_text: str = Field(..., description="Unstructured text from an invoice or receipt")

SYSTEM_PROMPT = """
You are a highly accurate data extraction assistant.
Analyze the provided invoice/receipt text and extract structured data.
You MUST respond ONLY with a valid raw JSON object matching this schema. Do NOT include markdown code blocks or any other text.
"""

@app.get("/")
def read_root():
    return {"status": "online", "message": "Free Invoice Extractor API is up and running."}

@app.post("/extract-invoice", response_model=InvoiceExtractionResponse)
def extract_invoice(payload: InvoiceExtractionRequest):
    if not payload.raw_text.strip():
        raise HTTPException(status_code=400, detail="Raw text cannot be empty.")

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Extract invoice data from this text:\n\n{payload.raw_text}"}
            ],
            temperature=0.1
        )
        
        extracted_data = json.loads(response.choices[0].message.content)
        return extracted_data

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse JSON output.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
