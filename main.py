import os
import json
import re
import shutil
import logging
from io import BytesIO
from typing import List, Optional, Union, Dict
import asyncio
import requests

from fastapi import FastAPI, File, Form, UploadFile, Request, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from bs4 import BeautifulSoup
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from playwright.async_api import async_playwright
from pydantic import BaseModel
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from sqlmodel import Field, Session, SQLModel, create_engine
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Environment Variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not found in environment variables")
    raise ValueError("OPENAI_API_KEY is required")

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///intake_sessions.db")
# Auto-correct PostgreSQL URLs if needed
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
logger.info(f"Using database: {DATABASE_URL}")

# Google Drive Configuration
SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE", "service_account.json")
if not os.path.exists(SERVICE_ACCOUNT_FILE):
    logger.error(f"Service account file not found: {SERVICE_ACCOUNT_FILE}")
    raise FileNotFoundError(f"Service account file not found: {SERVICE_ACCOUNT_FILE}")

GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
if not GOOGLE_DRIVE_FOLDER_ID:
    logger.error("GOOGLE_DRIVE_FOLDER_ID not found in environment variables")
    raise ValueError("GOOGLE_DRIVE_FOLDER_ID is required")

# Other configurations
SCOPES = ['https://www.googleapis.com/auth/drive.file']
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://localhost:8000",
    "https://stupendous-licorice-8114c0.netlify.app/",  # Add your Netlify domain
    os.getenv("FRONTEND_URL", "")
]
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
BRAND_GUIDELINES_DIR = os.getenv("BRAND_GUIDELINES_DIR", "brand_guidelines")
PDF_EXPORTS_DIR = os.getenv("PDF_EXPORTS_DIR", "pdf_exports")

# Create necessary directories
os.makedirs(BRAND_GUIDELINES_DIR, exist_ok=True)
os.makedirs(PDF_EXPORTS_DIR, exist_ok=True)

app = FastAPI(title="Creative Brief Generator API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)

engine = create_engine(DATABASE_URL, echo=False)
CACHE: Dict[str, dict] = {}

# --- MODELS ---
class SessionData(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    brand_input: str
    form_data: str
    website_data: str
    amazon_data: str
    brand_guideline_file_path: Optional[str] = None
    finalized: bool = Field(default=False)
    section_status: str = Field(default="{}")

# --- UTILITY FUNCTIONS ---
def sanitize_filename(filename: str) -> str:
    sanitized = re.sub(r'[\\/*?:"<>|]', "_", filename)
    sanitized = sanitized.strip(". ")
    if len(sanitized) > 200:
        sanitized = sanitized[:200]
    if not sanitized:
        sanitized = "unnamed_project"
    return sanitized

def save_uploaded_file(upload_file: UploadFile, save_dir=BRAND_GUIDELINES_DIR) -> str:
    safe_filename = sanitize_filename(upload_file.filename)
    save_path = os.path.join(save_dir, safe_filename)
    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        return save_path
    except Exception as e:
        logger.error(f"Error saving file {safe_filename}: {e}")
        return ""

def parse_gpt_output(gpt_output: str) -> dict:
    default_return = {"editable_sections": []}
    if not gpt_output or not gpt_output.strip():
        logger.warning("GPT output is empty!")
        return default_return
    try:
        data = json.loads(gpt_output)
        if isinstance(data, dict) and "editable_sections" in data and isinstance(data["editable_sections"], list):
            return data
        elif isinstance(data, list):
            logger.warning("GPT returned a list instead of dict, wrapping.")
            return {"editable_sections": data}
        else:
            logger.warning("GPT output JSON structure is not as expected.")
            return default_return
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing GPT JSON output: {e}")
        logger.error(f"Problematic GPT output (first 500 chars): {gpt_output[:500]}...")
        try:
            last_valid = max(gpt_output.rfind('}'), gpt_output.rfind(']'))
            if last_valid > 0:
                truncated = gpt_output[:last_valid+1]
                data = json.loads(truncated)
                logger.info("Successfully extracted partial valid JSON.")
                if isinstance(data, dict) and "editable_sections" in data:
                    return data
                elif isinstance(data, list):
                    return {"editable_sections": data}
        except Exception as inner_e:
            logger.error(f"Failed to extract partial JSON: {inner_e}")
        return {"editable_sections": [{"title": "Error", "questions": [{"question": "Parsing Failed", "answer": f"GPT output parsing error: {str(e)}"}]}]}
    except Exception as e:
        logger.error(f"Unexpected error parsing GPT output: {e}")
        return {"editable_sections": [{"title": "Error", "questions": [{"question": "Parsing Failed", "answer": f"Unexpected error: {str(e)}"}]}]}

# --- SCRAPING FUNCTIONS ---
async def scrape_website_text(url: str) -> str:
    logger.info(f"Attempting to scrape website: {url}")
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, timeout=15000)
            content = await page.content()
            soup = BeautifulSoup(content, 'html.parser')
            texts = [element.get_text(separator=' ', strip=True) for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'li', 'span'])]
            full_text = ' '.join(filter(None, texts))
            await browser.close()
            logger.info(f"Website scraping successful for {url}.")
            return full_text
    except Exception as e:
        logger.error(f"Could not scrape website {url}: {e}")
        return f"Could not scrape website: {e}"

async def scrape_amazon_listing_details(url: str) -> dict:
    logger.info(f"Attempting to scrape Amazon listing: {url}")
    details = {
        "title": "Not Found", "brand": "Not Found", "bullets": [],
        "description": "Not Found", "reviews_raw": [], "qna_raw": []
    }
    if url in CACHE:
        logger.info(f"Using cached data for Amazon URL: {url}")
        return CACHE[url]
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, timeout=60000)
            try:
                await page.wait_for_load_state("networkidle", timeout=40000)
                logger.info("Network idle state reached.")
            except Exception as timeout_err:
                logger.warning(f"Network idle timeout exceeded for {url}: {timeout_err}. Proceeding with available content.")
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(3)
            content = await page.content()
            soup = BeautifulSoup(content, 'html.parser')

            title_element = soup.select_one('#productTitle')
            details['title'] = title_element.get_text(strip=True) if title_element else "Title not found"

            brand_element = soup.select_one('#bylineInfo') or soup.select_one('a#brand')
            details['brand'] = brand_element.get_text(strip=True).replace('Visit the ', '').replace(' Store', '').replace('Brand: ', '') if brand_element else "Brand not found"

            bullet_elements = soup.select('#feature-bullets .a-list-item')
            details['bullets'] = [li.get_text(strip=True) for li in bullet_elements if li.get_text(strip=True)]

            desc_element = soup.select_one('#productDescription')
            details['description'] = desc_element.get_text(strip=True) if desc_element else "Description not found"

            review_elements = soup.select('div[data-hook="review-collapsed"] span, .review-text-content span')[:15]
            details['reviews_raw'] = [review.get_text(strip=True) for review in review_elements if review.get_text(strip=True)]

            qna_blocks = soup.select('#ask_lazy_load_div .a-fixed-left-grid-inner, div[data-cel-widget^="ask-"]')[:10]
            for block in qna_blocks:
                question_elem = block.select_one('a.a-link-normal[href*="ask/questions/"], .a-declarative .a-link-normal span')
                answer_elem = block.select_one('.a-col-right .a-row span, span[class*="ask-answer-"]')
                if question_elem and answer_elem:
                    q = question_elem.get_text(strip=True)
                    a = answer_elem.get_text(strip=True)
                    if q and a:
                        details['qna_raw'].append(f"Q: {q} A: {a}")

            await browser.close()
            CACHE[url] = details
            logger.info("Amazon scraping successful with Playwright.")
            return details
    except Exception as e:
        logger.error(f"Error during Playwright Amazon scraping for {url}: {e}. Falling back to requests method.")
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                title_element = soup.select_one('#productTitle')
                details['title'] = title_element.get_text(strip=True) if title_element else "Title not found"
                brand_element = soup.select_one('#bylineInfo') or soup.select_one('a#brand')
                details['brand'] = brand_element.get_text(strip=True).replace('Visit the ', '').replace(' Store', '').replace('Brand: ', '') if brand_element else "Brand not found"
                bullet_elements = soup.select('#feature-bullets .a-list-item')
                details['bullets'] = [li.get_text(strip=True) for li in bullet_elements if li.get_text(strip=True)]
                desc_element = soup.select_one('#productDescription')
                details['description'] = desc_element.get_text(strip=True) if desc_element else "Description not found"
                logger.info("Amazon scraping successful with fallback requests method.")
            else:
                logger.warning(f"Fallback requests method failed with status code {response.status_code} for {url}.")
            CACHE[url] = details
            return details
        except Exception as req_e:
            logger.error(f"Error during fallback requests scraping for {url}: {req_e}")
            return details

# --- GPT UTILITY FUNCTIONS ---
@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((Exception))
)
def call_openai_api(prompt: str, model: str = GPT_MODEL, temperature: float = 0.5, response_format: dict = None) -> str:
    logger.info("Calling OpenAI API...")
    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are an expert Amazon strategist."},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": 4096
        }

        if response_format:
            payload["response_format"] = response_format

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )

        if response.status_code != 200:
            logger.error(f"OpenAI API returned status code {response.status_code}: {response.text}")
            raise Exception(f"OpenAI API error: {response.text}")

        response_data = response.json()
        content = response_data["choices"][0]["message"]["content"]
        logger.info("OpenAI API call successful.")
        return content
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        if "token" in str(e).lower() and len(prompt) > 4000:
            logger.info("Token limit likely exceeded, retrying with shorter prompt.")
            shortened_prompt = prompt[:4000] + "\n[Content truncated due to length. Please summarize based on available data.]"

            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are an expert Amazon strategist."},
                    {"role": "user", "content": shortened_prompt}
                ],
                "temperature": temperature,
                "max_tokens": 4096
            }

            if response_format:
                payload["response_format"] = response_format

            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )

            if response.status_code != 200:
                logger.error(f"OpenAI API returned status code {response.status_code}: {response.text}")
                raise Exception(f"OpenAI API error: {response.text}")

            response_data = response.json()
            content = response_data["choices"][0]["message"]["content"]
            logger.info("Retry with shorter prompt successful.")
            return content
        raise

def extract_and_transform_voc(reviews: List[str], qna: List[str], top_n: int = 3) -> str:
    full_text = "\n".join(reviews + qna)
    if not full_text.strip():
        return "No customer reviews or Q&A found to analyze."
    scraped_info = full_text[:4000]
    prompt = f"""
    Based *only* on the following scraped data from an Amazon product page, generate a comprehensive creative brief.

    **Scraped Data:**
    ```
    {scraped_info}
    ```

    **Instructions:**
    Your task is to create a creative brief containing exactly the 16 sections listed below.
    You MUST use the provided "Scraped Data" as your *sole* source of information. Do not invent features, benefits, or target audiences not supported by the data.
    **Crucially, you MUST fill in EVERY section.**

    If the scraped data doesn't directly provide information for a specific section:
    1.  **Infer Logically:** Make reasonable inferences *strictly* based on the available product details (title, category implied by title/description, features, price). For example, infer 'Brand' from the title if possible. Infer 'Product Category' based on title/description.
    2.  **State Assumptions/Limitations:** If inferring, clearly state it (e.g., "Inferred Target Audience:", "Assumed Brand:"). If data is completely missing for a section (e.g., no features listed), explicitly state that the information was not available in the provided scrape (e.g., "Key Features: No specific features were listed in the provided data."). Do *not* make up features.
    3.  **Suggest Next Steps (If Applicable):** For sections like 'Competitor Awareness' or 'Target Audience' where data is often sparse, state that further research is needed (e.g., "Competitor Awareness: Research required to identify key competitors for [Product Name/Category].", "Target Audience: Based on the product type ([Category]), the likely audience is [General Group], but detailed persona development is recommended.").
    4.  **Do NOT leave any section blank.** Provide *some* relevant content for every point, even if it's just stating the lack of specific data and suggesting research.
    5.  **Format the output as a single JSON object** where keys are the exact section titles (e.g., "Project Title", "Brand", etc.) and values are the generated text content for that section. Ensure the output is valid JSON.

    **16 Required Sections (Use these exact keys in the JSON):**
    1.  **Project Title:** (Generate a title like "Creative Brief for [Product Name]")
    2.  **Brand:** (Extract or infer the brand name from the title or description. State if inferred or if brand is not specified in the data.)
    3.  **Product Name:** (Use the exact 'Product Title' from scraped data, or state if not found.)
    4.  **Product Category:** (Infer a likely category based on title/description. State the inferred category.)
    5.  **Executive Summary:** (Provide a 2-3 sentence overview summarizing the product based *only* on the scraped data and the purpose of this brief.)
    6.  **Detailed Product Description:** (Synthesize a description using the 'Description' and 'Features' from scraped data. If description is missing, summarize features or state lack of descriptive data.)
    7.  **Key Features:** (List the features from the 'Features' section of scraped data. If empty or 'N/A', state "No specific features were listed in the provided data.")
    8.  **Key Benefits:** (Attempt to translate listed features into customer benefits. If no features, state "Benefits cannot be determined as no key features were provided in the data.")
    9.  **Target Audience:** (Infer a *general* target audience based on product title/category. State this is an inference and recommend further research, e.g., "Inferred General Audience: [General description based on product type]. Detailed audience research recommended.")
    10. **Unique Selling Proposition (USP):** (Identify a potential USP *if supported by the data* (e.g., very high rating, specific unique feature listed, significantly low price if context available). If not clear, state "USP not readily apparent from the provided data; further analysis or brand input needed.")
    11. **Competitor Awareness:** (State "Competitor information not available in the scraped data. Market research recommended.")
    12. **Marketing Objective:** (Suggest a plausible primary objective based on the product type, e.g., "Primary Objective Suggestion: Drive sales conversions on the Amazon platform.")
    13. **Key Message:** (Propose a concise message *based only on available data*, e.g., "[Product Name]: [Highlight a key feature or inferred benefit, if available]." If data is sparse, suggest "Key message development requires further input on USP and target audience.")
    14. **Tone of Voice:** (Suggest a tone based on product category/price/title, e.g., "Suggested Tone: [Informative/Playful/Premium], based on the product being a [category/type]. Tone should be finalized based on brand guidelines.")
    15. **Mandatories/Constraints:** (List any obvious constraints from data, e.g., "Must adhere to Amazon listing policies." If none apparent, state "No specific mandatories identified in the scraped data.")
    16. **Call to Action (CTA):** (Suggest standard e-commerce CTAs, e.g., "Suggested CTAs: 'Add to Cart', 'Buy Now', 'Learn More on Amazon'.")

    **Output Format:** Respond ONLY with the valid JSON object containing these 16 sections. Do not include any explanatory text before or after the JSON.
    """
    try:
        return call_openai_api(prompt, temperature=0.6)
    except Exception as e:
        logger.error(f"Error generating VOC insights: {e}")
        return f"Error generating VOC insights: {e}"

def generate_creative_brief(inputs: dict, website_data: str, amazon_details: dict, voc_insights: str, brand_guideline_file_path: Optional[str] = None) -> str:
    amazon_title = amazon_details.get("title", "N/A")
    amazon_brand = amazon_details.get("brand", "N/A")
    amazon_bullets_str = "\n- ".join(amazon_details.get("bullets", []))
    amazon_bullets_str = "- " + amazon_bullets_str if amazon_bullets_str else "N/A"
    amazon_description = amazon_details.get("description", "N/A")

    website_section = f"### Scraped Website Content (Supplementary - Use only if needed):\n{website_data[:2000]}..." if website_data else "### Scraped Website Content (Supplementary - Use only if needed):\nNo website URL provided or scraping failed."
    guideline_section = f"*Brand Guideline File Uploaded:* {os.path.basename(brand_guideline_file_path)}" if brand_guideline_file_path else ""

    user_input_summary_items = []
    for k, v in inputs.items():
        if k not in ['amazon_listing', 'website_url', 'brand_guideline_file']:
            if isinstance(v, list):
                user_input_summary_items.append(f"- {k}: {', '.join(v) if v else 'N/A'}")
            else:
                user_input_summary_items.append(f"- {k}: {v if v else 'N/A'}")
    user_input_summary = "\n".join(user_input_summary_items)

    prompt = f"""
    You are an expert Amazon strategist and copywriter generating a Creative Brief JSON.

    **Primary Data Source:** Scraped Amazon Listing Data provided below.
    **Secondary Data Source:** User Provided Input (confirm details, fill gaps).
    **Tertiary Data Source:** Scraped Website Content (supplementary context only).

    **CRITICAL INSTRUCTIONS:**
    1. Generate the **complete** Creative Brief in the specified JSON format.
    2. Answer **ALL** questions thoroughly with detailed, product-related content.
    3. Derive answers **PRIMARILY** from the "Scraped Amazon Listing Data". Use the Title, Brand, Bullets, Description, and Customer Insights (VOC).
    4. Use "User Provided Input" to confirm information (like brand name, product category) or fill details not explicitly on the Amazon page (like specific competitors, detailed brand story).
    5. Use "Scraped Website Content" **only** if information is missing from both Amazon and User Input.
    6. **DO NOT LEAVE ANSWERS BLANK.** If information isn't directly stated in the Amazon data or other sources, **infer logically** based on any available context (e.g., product title, category, or general user input). If minimal data is available, **generate detailed content** related to the product by making educated assumptions and clearly state these as suggestions (e.g., "Suggested Target Audience based on product type: [Detailed description]. Further research recommended."). Provide actionable insights and creative suggestions to fill all fields comprehensively.
    7. Ensure all responses are **detailed and specific**, avoiding generic or vague answers. For each section, elaborate on how the information or suggestion ties to the product or brand strategy on Amazon.
    8. Output **ONLY** the valid JSON structure. No introductory text, comments, or markdown formatting outside the JSON values.

    ---
    ### Scraped Amazon Listing Data (PRIMARY SOURCE):
    **Brand:** {amazon_brand}
    **Product Title:** {amazon_title}
    **Bullet Points:**
    {amazon_bullets_str}
    **Product Description:**
    {amazon_description[:2500]}...
    **Customer Insights (VOC from Reviews/Q&A):**
    {voc_insights}
    ---
    ### User Provided Input (Secondary Source):
    {user_input_summary}
    {guideline_section}
    ---
    {website_section}
    ---

    **JSON Structure to Generate:**
    {{"editable_sections": [
        {{"title": "PROJECT OVERVIEW", "questions": [
            {{"question": "Project Name", "answer": ""}},
            {{"question": "Brand Name", "answer": ""}},
            {{"question": "Website", "answer": ""}},
            {{"question": "Amazon Listing (if available)", "answer": "{inputs.get('amazon_listing', '')}"}},
            {{"question": "Instagram Handle (if applicable)", "answer": ""}}
        ]}},
        {{"title": "PRODUCT SNAPSHOT", "questions": [
            {{"question": "What exactly is the product?", "answer": ""}},
            {{"question": "What does it do and how does it work?", "answer": ""}},
            {{"question": "What problem does it solve?", "answer": ""}},
            {{"question": "Who is it meant for?", "answer": ""}}
        ]}},
        {{"title": "CURRENT LISTING CHALLENGES", "questions": [
            {{"question": "What's broken or underwhelming about the current Amazon listing, brand positioning, or creative execution?", "answer": ""}},
            {{"question": "Where are they losing conversions or attention?", "answer": ""}}
        ]}},
        {{"title": "TARGET CUSTOMER DEEP DIVE", "questions": [
            {{"question": "Gender, age range, location, income, profession", "answer": ""}},
            {{"question": "Life stage or identity (e.g., new moms, eco-conscious Gen Z, busy professionals)", "answer": ""}},
            {{"question": "Pain points, desires, motivations", "answer": ""}},
            {{"question": "How do they shop on Amazon? What do they care about when scrolling?", "answer": ""}}
        ]}},
        {{"title": "BARRIERS TO PURCHASE", "questions": [
            {{"question": "List the common doubts, hesitations, or FAQ-style friction points that stop people from buying — even if they like the product.", "answer": ""}}
        ]}},
        {{"title": "BRAND VOICE & TONE", "questions": [
            {{"question": "Describe the tone and copywriting style the brand uses or should use (e.g., bold, sassy, informative, premium, conversational).", "answer": ""}},
            {{"question": "Include any signature words, phrases, or linguistic quirks.", "answer": ""}}
        ]}},
        {{"title": "USPs (UNIQUE SELLING PROPOSITIONS)", "questions": [
            {{"question": "What makes this product meaningfully different from other options in the category?", "answer": ""}},
            {{"question": "Think functional benefits, emotional angles, and cultural relevance.", "answer": ""}}
        ]}},
        {{"title": "5-SECOND WOW FACTOR", "questions": [
            {{"question": "If a customer saw this listing for 5 seconds, what single visual hook, copy line, or feature would stop them in their tracks?", "answer": ""}}
        ]}},
        {{"title": "KEY FEATURES (WITH CONTEXT)", "questions": [
            {{"question": "List 4–6 major features. But go beyond just the bullet points — explain: Why does this matter to the buyer? How does it connect to their lifestyle or values?", "answer": ""}}
        ]}},
        {{"title": "TOP 6 SELLING POINTS (WITH STRATEGIC JUSTIFICATION)", "questions": [
            {{"question": "For each of the client's selected selling points: State the point. Explain *why* it's strategically powerful for this product and customer.", "answer": ""}}
        ]}},
        {{"title": "COMPETITIVE LANDSCAPE", "questions": [
            {{"question": "List 2–3 main competitors", "answer": ""}},
            {{"question": "Describe how this product compares", "answer": ""}},
            {{"question": "Mention any Amazon-specific differentiators (e.g. bundle, shipping time, design)", "answer": ""}}
        ]}},
        {{"title": "SEARCH & KEYWORDS STRATEGY", "questions": [
            {{"question": "Suggest relevant search terms and niche keywords to target. These should align with user intent, category trends, or long-tail SEO goals.", "answer": ""}}
        ]}},
        {{"title": "BRAND STORY, VALUES & PURPOSE", "questions": [
            {{"question": "Give a short but meaningful brand origin story or founder story.", "answer": ""}},
            {{"question": "Highlight core values, emotional drivers, or the \"bigger why\" behind the brand's existence.", "answer": ""}}
        ]}},
        {{"title": "DESIGN DIRECTION", "questions": [
            {{"question": "Summarize the client's aesthetic preferences", "answer": ""}},
            {{"question": "Suggest how the visuals, layout, or color themes should feel (e.g., clean/minimal, bold/graphic, warm/natural)", "answer": ""}}
        ]}},
        {{"title": "FINAL NOTES & STRATEGIC CALLOUTS", "questions": [
            {{"question": "Include any extra insights for the creative team, such as: Packaging or compliance considerations, Customer education needs, Cross-sell or upsell potential, Social proof or influencer angles", "answer": ""}}
        ]}}
    ]}}

    Generate the JSON output now:
    """
    try:
        return call_openai_api(prompt, response_format={"type": "json_object"})
    except Exception as e:
        logger.error(f"Error generating creative brief: {e}")
        return json.dumps({"editable_sections": [{"title": "Error", "questions": [{"question": "Generation Failed", "answer": str(e)}]}]})

# --- BACKGROUND TASKS ---
def generate_pdf_in_background(session_id: int, project_name: str):
    logger.info(f"Generating PDF for session {session_id} in background.")
    with Session(engine) as db:
        session = db.get(SessionData, session_id)
        if not session:
            logger.error(f"Session {session_id} not found for PDF generation.")
            return
        safe_filename_base = sanitize_filename(project_name)
        pdf_filename = f"{safe_filename_base}.pdf"
        pdf_path = os.path.join(PDF_EXPORTS_DIR, pdf_filename)
        try:
            doc = SimpleDocTemplate(pdf_path, pagesize=LETTER, rightMargin=inch*0.75, leftMargin=inch*0.75, topMargin=inch*0.75, bottomMargin=inch*0.75)
            story = []
            styles = getSampleStyleSheet()
            h1_style = ParagraphStyle(name='H1Style', parent=styles['h1'], fontSize=16, spaceAfter=14, fontName='Helvetica-Bold')
            h2_style = ParagraphStyle(name='H2Style', parent=styles['h2'], fontSize=12, spaceAfter=10, spaceBefore=10, fontName='Helvetica-Bold')
            question_style = ParagraphStyle(name='QuestionStyle', parent=styles['Normal'], fontSize=10, fontName='Helvetica-Bold', spaceAfter=2)
            answer_style = ParagraphStyle(name='AnswerStyle', parent=styles['Normal'], fontSize=10, fontName='Helvetica', leftIndent=10, spaceAfter=10)
            body_style = ParagraphStyle(name='BodyStyle', parent=styles['Normal'], fontSize=10, leading=14, spaceAfter=10, fontName='Helvetica', alignment=TA_LEFT)

            story.append(Paragraph(f"Creative Brief: {project_name}", h1_style))
            story.append(Spacer(1, 0.2 * inch))

            inputs = json.loads(session.brand_input) if session.brand_input else {}
            story.append(Paragraph("ASSETS OVERVIEW", h2_style))
            if inputs.get("has_assets", True):
                asset_fields = [
                    ("White Background Image Link", "white_background_image"),
                    ("Old Images Link", "old_images"),
                    ("Lifestyle Image Link", "lifestyle_image"),
                    ("User-Generated Content Link", "user_generated_content"),
                    ("Video Content Link", "video_content"),
                ]
                any_asset = False
                for label, key in asset_fields:
                    value = inputs.get(key, "")
                    if value:
                        any_asset = True
                        story.append(Paragraph(f"<b>{label}:</b> {value}", body_style))
                if not any_asset:
                    story.append(Paragraph("No asset links provided.", body_style))
            else:
                story.append(Paragraph("No assets provided.", body_style))

            sections = json.loads(session.form_data) if session.form_data else []
            for section in sections:
                heading = section.get("title", "Untitled Section")
                story.append(Paragraph(heading.upper(), h2_style))
                if "questions" in section and isinstance(section["questions"], list):
                    for qa in section["questions"]:
                        q = qa.get("question", "")
                        a = qa.get("answer", "N/A")
                        if q:
                            story.append(Paragraph(q, question_style))
                            story.append(Paragraph(a.replace("\n", "<br />"), answer_style))
                story.append(Spacer(1, 0.1 * inch))

            doc.build(story)
            logger.info(f"PDF generated at {pdf_path} for session {session_id}.")
        except Exception as e:
            logger.error(f"Error generating PDF for session {session_id}: {e}")

def upload_to_drive(file_path: str, filename: str):
    try:
        creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        service = build('drive', 'v3', credentials=creds)
        file_metadata = {'name': filename, 'parents': [GOOGLE_DRIVE_FOLDER_ID]}
        media = MediaFileUpload(file_path, mimetype='application/pdf', resumable=True)
        logger.info(f"Uploading {filename} to Google Drive...")
        file = service.files().create(body=file_metadata, media_body=media, fields='id, webViewLink').execute()
        logger.info(f"File uploaded. ID: {file.get('id')}")
        return file.get('id'), file.get('webViewLink')
    except Exception as e:
        logger.error(f"Error uploading to Google Drive: {e}")
        raise

# --- FASTAPI ROUTES ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unexpected error occurred: {str(exc)}")
    return JSONResponse(status_code=500, content={"detail": f"An unexpected error occurred: {str(exc)}"})

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTP error occurred: {str(exc.detail)}")
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

@app.on_event("startup")
async def on_startup():
    SQLModel.metadata.create_all(engine)
    os.makedirs(BRAND_GUIDELINES_DIR, exist_ok=True)
    os.makedirs(PDF_EXPORTS_DIR, exist_ok=True)
    logger.info("Application startup complete.")

@app.post("/collect-input")
async def collect_input(
    project_name: str = Form(...), brand_name: str = Form(...), product_about: str = Form(...),
    amazon_listing: str = Form(""), website_url: str = Form(""), instagram_url: str = Form(""),
    listing_elements: Union[List[str], str] = Form([]), product_category: Union[List[str], str] = Form([]),
    current_product_info_link: str = Form(""), whats_not_working: Union[List[str], str] = Form([]),
    whats_not_working_other: str = Form(""), go_beyond_brand_guide: str = Form(""),
    brand_guideline_file: Optional[UploadFile] = File(None), has_assets: bool = Form(True),
    white_background_image: str = Form(""), old_images: str = Form(""),
    lifestyle_image: str = Form(""), user_generated_content: str = Form(""),
    video_content: str = Form(""), no_assets_text: str = Form(""),
    background_tasks: BackgroundTasks = None
):
    logger.info("Received input collection request.")
    def ensure_list(val):
        if isinstance(val, str):
            return [item.strip() for item in val.split(',') if item.strip()] if val else []
        return val if isinstance(val, list) else []
    listing_elements = ensure_list(listing_elements)
    product_category = ensure_list(product_category)
    whats_not_working = ensure_list(whats_not_working)
    file_path = save_uploaded_file(brand_guideline_file) if brand_guideline_file else None
    inputs = {
        "project_name": project_name, "brand_name": brand_name, "product_about": product_about,
        "amazon_listing": amazon_listing, "website_url": website_url, "instagram_url": instagram_url,
        "listing_elements": listing_elements, "product_category": product_category,
        "current_product_info_link": current_product_info_link, "whats_not_working": whats_not_working,
        "whats_not_working_other": whats_not_working_other, "go_beyond_brand_guide": go_beyond_brand_guide,
        "has_assets": has_assets, "white_background_image": white_background_image,
        "old_images": old_images, "lifestyle_image": lifestyle_image,
        "user_generated_content": user_generated_content, "video_content": video_content,
        "no_assets_text": no_assets_text
    }
    amazon_details = await scrape_amazon_listing_details(amazon_listing) if amazon_listing else {}
    voc_insights = extract_and_transform_voc(amazon_details.get('reviews_raw', []), amazon_details.get('qna_raw', [])) if amazon_details else "No Amazon data."
    website_data = await scrape_website_text(website_url) if website_url else ""
    gpt_output_str = generate_creative_brief(inputs, website_data, amazon_details, voc_insights, file_path)
    form_json_sections = parse_gpt_output(gpt_output_str).get("editable_sections", [])
    section_status = {str(i): {"saved": False, "approved": False} for i in range(len(form_json_sections))}
    amazon_data_to_store = json.dumps({"scraped_details": amazon_details, "voc_insights": voc_insights})
    with Session(engine) as db:
        session = SessionData(
            brand_input=json.dumps(inputs), form_data=json.dumps(form_json_sections),
            website_data=website_data, amazon_data=amazon_data_to_store,
            brand_guideline_file_path=file_path, section_status=json.dumps(section_status)
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        logger.info(f"Session {session.id} created successfully.")
    return {"session_id": session.id, "form": form_json_sections}

@app.get("/get-editable-form/{session_id}")
def get_editable_form(session_id: int):
    with Session(engine) as db:
        session = db.get(SessionData, session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        section_status = json.loads(session.section_status) if session.section_status else {}
        form_sections = json.loads(session.form_data) if session.form_data else []
        return {
            "editable_sections": form_sections,
            "finalized": session.finalized,
            "section_status": section_status
        }
@app.get("/")
async def root():
    return {"status": "online", "message": "Creative Brief Generator API is running", "docs_url": "/docs"}
class EditedFormInput(BaseModel):
    session_id: int
    sections: List[dict]

@app.post("/save-edited-form")
def save_edited_form(data: EditedFormInput):
    with Session(engine) as db:
        session = db.get(SessionData, data.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        if session.finalized:
            raise HTTPException(status_code=403, detail="Form already finalized")
        session.form_data = json.dumps(data.sections)
        db.add(session)
        db.commit()
        logger.info(f"Full form saved for session {data.session_id}.")
        return {"message": "Form updated successfully."}

class SectionInput(BaseModel):
    session_id: int
    section_index: int
    section_data: dict

@app.post("/save-section")
def save_section(data: SectionInput):
    with Session(engine) as db:
        session = db.get(SessionData, data.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        if session.finalized:
            raise HTTPException(status_code=403, detail="Form already finalized")
        sections = json.loads(session.form_data) if session.form_data else []
        if not (0 <= data.section_index < len(sections)):
            raise HTTPException(status_code=400, detail="Invalid section index")
        sections[data.section_index] = data.section_data
        session.form_data = json.dumps(sections)
        section_status = json.loads(session.section_status) if session.section_status else {}
        current_status = section_status.get(str(data.section_index), {"saved": False, "approved": False})
        current_status["saved"] = True
        section_status[str(data.section_index)] = current_status
        session.section_status = json.dumps(section_status)
        db.add(session)
        db.commit()
        logger.info(f"Section {data.section_index} saved for session {data.session_id}.")
        return {"message": "Section saved successfully."}

class SectionStatusInput(BaseModel):
    session_id: int
    section_index: int
    status: dict

@app.post("/update-section-status")
def update_section_status(data: SectionStatusInput):
    with Session(engine) as db:
        session = db.get(SessionData, data.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        if session.finalized:
            raise HTTPException(status_code=403, detail="Form already finalized")
        section_status = json.loads(session.section_status) if session.section_status else {}
        if str(data.section_index) not in section_status:
            section_status[str(data.section_index)] = {"saved": False, "approved": False}
        section_status[str(data.section_index)]["saved"] = data.status.get("saved", section_status[str(data.section_index)]["saved"])
        section_status[str(data.section_index)]["approved"] = data.status.get("approved", section_status[str(data.section_index)]["approved"])
        session.section_status = json.dumps(section_status)
        db.add(session)
        db.commit()
        logger.info(f"Section {data.section_index} status updated for session {data.session_id}: {data.status}")
        return {"message": "Section status updated successfully."}

@app.post("/approve-final-form/{session_id}")
def approve_final_form(session_id: int):
    with Session(engine) as db:
        session = db.get(SessionData, session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        session.finalized = True
        db.add(session)
        db.commit()
        logger.info(f"Form finalized for session {session_id}.")
        return {"message": "Form approved and locked."}

@app.post("/export-form-pdf/{session_id}")
async def export_form_pdf(session_id: int, background_tasks: BackgroundTasks):
    with Session(engine) as db:
        session = db.get(SessionData, session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        inputs = json.loads(session.brand_input) if session.brand_input else {}
        project_name = inputs.get("project_name", f"project_{session_id}")
        background_tasks.add_task(generate_pdf_in_background, session_id, project_name)
        return {"message": "PDF generation started in background."}

@app.post("/upload-pdf-to-drive/{session_id}")
def upload_pdf_to_drive_endpoint(session_id: int):
    with Session(engine) as db:
        session = db.get(SessionData, session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        inputs = json.loads(session.brand_input) if session.brand_input else {}
        project_name = inputs.get("project_name", f"project_{session_id}")
        safe_filename_base = sanitize_filename(project_name)
        pdf_filename = f"{safe_filename_base}.pdf"
        pdf_path = os.path.join(PDF_EXPORTS_DIR, pdf_filename)
        if not os.path.exists(pdf_path):
            raise HTTPException(status_code=404, detail="PDF not found. Generate it first.")
        file_id, preview_url = upload_to_drive(pdf_path, pdf_filename)
        return {"message": "Uploaded to Google Drive", "file_id": file_id, "preview_url": preview_url}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting FastAPI server on port {port}...")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)