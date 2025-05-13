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
SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE", "/opt/render/project/src/service_account.json")
if not os.path.exists(SERVICE_ACCOUNT_FILE):
    logger.error(f"Service account file not found: {SERVICE_ACCOUNT_FILE}")
    raise FileNotFoundError(f"Service account file not found: {SERVICE_ACCOUNT_FILE}")

GOOGLE_DRIVE_FOLDER_ID = "17YHVHs1oRA3fj9iPcN5hIo0FXXm_XB8W"
if not GOOGLE_DRIVE_FOLDER_ID:
    logger.error("GOOGLE_DRIVE_FOLDER_ID not found in environment variables")
    raise ValueError("GOOGLE_DRIVE_FOLDER_ID is required")

# Other configurations
SCOPES = ['https://www.googleapis.com/auth/drive.file']
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://localhost:8000",
    "https://stupendous-licorice-8114c0.netlify.app",
    "https://onboarding.workinxdigital.com",  # Removed trailing slash
    os.getenv("FRONTEND_URL", "")
]
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
BRAND_GUIDELINES_DIR = os.getenv("BRAND_GUIDELINES_DIR", "brand_guidelines")
PDF_EXPORTS_DIR = os.getenv("PDF_EXPORTS_DIR", "pdf_exports")

# Initialize OpenAI
import openai
openai.api_key = OPENAI_API_KEY

# Create necessary directories
os.makedirs(BRAND_GUIDELINES_DIR, exist_ok=True)
os.makedirs(PDF_EXPORTS_DIR, exist_ok=True)


app = FastAPI(title="Creative Brief Generator API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # Added OPTIONS
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

            # Check if scraping returned minimal content
            if len(full_text) < 100:
                logger.warning(f"Website scraping returned minimal content for {url}. Will try fallback.")
                raise Exception("Minimal content retrieved")

            return full_text
    except Exception as e:
        logger.error(f"Could not scrape website {url}: {e}")

        # Check if scraping failed or returned minimal content
        try:
            logger.info(f"Website scraping failed, using enhanced web search fallback for: {url}")
            # Create a query from the URL
            domain = url.split("//")[-1].split("/")[0]

            # Try to extract product name from URL path
            path_parts = url.split("/")
            product_hint = ""
            if len(path_parts) > 3:
                product_hint = path_parts[-1].replace("-", " ").replace(".html", "").replace(".php", "")

            query = f"{domain} {product_hint} product information"

            # Use the enhanced fallback to get website content
            fallback_result = await enhanced_Web_Search_fallback(query)

            # Extract text content from the fallback result
            fallback_text = f"Title: {fallback_result['title']}\n"
            fallback_text += f"Brand: {fallback_result['brand']}\n"
            fallback_text += "Features:\n" + "\n".join([f"- {bullet}" for bullet in fallback_result['bullets']]) + "\n"
            fallback_text += f"Description: {fallback_result['description']}\n\n"

            # Add reviews if available
            if fallback_result['reviews_raw']:
                fallback_text += "Customer Feedback:\n" + "\n".join([f"- {review}" for review in fallback_result['reviews_raw'][:3]])

            logger.info("Successfully retrieved website content via enhanced web search fallback")
            return fallback_text
        except Exception as fallback_err:
            logger.error(f"Enhanced web search fallback for website content failed: {fallback_err}")
            return f"Could not scrape website: {e}. Web search fallback also failed: {fallback_err}"
async def enhanced_Web_Search_fallback(query: str, product_info: dict = None, amazon_url: str = None) -> dict:
    """
    Enhanced fallback function that uses a combination of web search and GPT-4
    to gather comprehensive product information when scraping methods fail.

    This implementation:
    1. Specifically searches for the Amazon listing URL if provided
    2. Uses targeted queries to gather different aspects of product information
    3. Synthesizes information into a structured format

    Args:
        query: Search query about the product
        product_info: Any partial information already collected
        amazon_url: The specific Amazon listing URL that failed to scrape properly

    Returns:
        dict: Completed product information
    """
    logger.info(f"Using enhanced web search fallback for: {query}")

    # Initialize with any existing data or empty structure
    details = product_info or {
        "title": "Not Found", "brand": "Not Found", "bullets": [],
        "description": "Not Found", "reviews_raw": [], "qna_raw": []
    }

    try:
        # Step 1: Perform targeted searches
        search_results = []
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }

        # First search: Direct Amazon URL search if available
        if amazon_url:
            logger.info(f"Searching specifically for Amazon listing: {amazon_url}")
            amazon_specific_query = f"product information from {amazon_url}"

            payload = {
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": "You are a helpful web search assistant."},
                    {"role": "user", "content": f"Search for: {amazon_specific_query}"}
                ],
                "tools": [{"type": "Web Search"}],
                "tool_choice": {"type": "Web Search"}
            }

            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )

            if response.status_code == 200:
                response_data = response.json()
                tool_calls = response_data["choices"][0]["message"].get("tool_calls", [])
                if tool_calls:
                    search_results.append(tool_calls[0]["function"].get("arguments", "{}"))

        # Second search: General product information if we don't have enough yet
        if not amazon_url or details["title"] == "Not Found" or not details["bullets"]:
            general_query = f"{query} product details specifications"
            payload = {
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": "You are a helpful web search assistant."},
                    {"role": "user", "content": f"Search for: {general_query}"}
                ],
                "tools": [{"type": "Web Search"}],
                "tool_choice": {"type": "Web Search"}
            }

            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )

            if response.status_code == 200:
                response_data = response.json()
                tool_calls = response_data["choices"][0]["message"].get("tool_calls", [])
                if tool_calls:
                    search_results.append(tool_calls[0]["function"].get("arguments", "{}"))

        # Third search: Reviews and customer feedback
        if details["reviews_raw"] == []:
            # If we have an Amazon URL, specifically search for reviews of that product
            reviews_query = f"{amazon_url} customer reviews" if amazon_url else f"{query} product reviews customer feedback"
            payload["messages"][1]["content"] = f"Search for: {reviews_query}"

            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )

            if response.status_code == 200:
                response_data = response.json()
                tool_calls = response_data["choices"][0]["message"].get("tool_calls", [])
                if tool_calls:
                    search_results.append(tool_calls[0]["function"].get("arguments", "{}"))

        # Fourth search: Q&A and common questions
        if details["qna_raw"] == []:
            # If we have an Amazon URL, specifically search for Q&A of that product
            qna_query = f"{amazon_url} questions and answers" if amazon_url else f"{query} common questions answers FAQ"
            payload["messages"][1]["content"] = f"Search for: {qna_query}"

            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )

            if response.status_code == 200:
                response_data = response.json()
                tool_calls = response_data["choices"][0]["message"].get("tool_calls", [])
                if tool_calls:
                    search_results.append(tool_calls[0]["function"].get("arguments", "{}"))

        # Step 2: Process search results with GPT-4 to extract structured information
        combined_search_results = "\n\n".join([json.loads(result).get("search_results", "") for result in search_results if result])

        # Include the Amazon URL in the extraction prompt if available
        amazon_url_context = f"Amazon Product URL: {amazon_url}\n\n" if amazon_url else ""

        extraction_prompt = f"""
        {amazon_url_context}Based on the following web search results about a product, extract structured information in the requested format.

        WEB SEARCH RESULTS:
        {combined_search_results}

        Extract the following information:
        1. Product Title (full and accurate)
        2. Brand Name
        3. Key Features/Bullet Points (list at least 5 if available)
        4. Product Description (detailed, at least 100 words)
        5. Sample Customer Reviews (3-5 if available)
        6. Common Questions and Answers about the product (2-3 if available)

        For any information you can't find with high confidence, indicate "Not Found" rather than guessing.
        Format your response as a valid JSON object with these keys: title, brand, bullets (array), description, reviews_raw (array), qna_raw (array).

        Ensure the JSON is properly formatted and valid. For reviews_raw and qna_raw, include full text entries, not summaries.
        """

        # Call OpenAI with JSON response format
        extraction_response = call_openai_api(
            prompt=extraction_prompt,
            model=GPT_MODEL,
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        # Parse the response
        extracted_data = json.loads(extraction_response)

        # Step 3: Update our details with any new information found
        for key, value in extracted_data.items():
            if key in details:
                # Only update if we don't have the data or it's a placeholder
                if details[key] == "Not Found" or (isinstance(details[key], list) and not details[key]):
                    details[key] = value
                # For text fields, prefer longer, more detailed content
                elif isinstance(value, str) and len(value) > len(details[key]) and details[key] != "Not Found":
                    details[key] = value
                # For lists, merge if both have content
                elif isinstance(value, list) and isinstance(details[key], list) and value and details[key]:
                    combined = details[key] + value
                    # Remove duplicates while preserving order
                    seen = set()
                    details[key] = [x for x in combined if not (x in seen or seen.add(x))]

        logger.info("Enhanced web search fallback successful")
        return details

    except Exception as e:
        logger.error(f"Enhanced web search fallback failed: {e}")
        # If the enhanced method fails, try a simpler approach
        try:
            # Include Amazon URL in the simple prompt if available
            amazon_context = f"from this Amazon listing: {amazon_url}" if amazon_url else ""

            simple_prompt = f"""
            I need information about this product: {query} {amazon_context}

            Please provide the following details in a structured format:
            1. Product Title
            2. Brand Name
            3. Key Features (list format)
            4. Product Description
            5. Sample Customer Reviews (if available)
            6. Common Questions and Answers (if available)

            Format as JSON with these keys: title, brand, bullets (array), description, reviews_raw (array), qna_raw (array).
            """

            simple_response = call_openai_api(
                prompt=simple_prompt,
                model=GPT_MODEL,
                temperature=0.3,
                response_format={"type": "json_object"}
            )

            simple_data = json.loads(simple_response)

            # Update our details with any new information found
            for key, value in simple_data.items():
                if key in details and (details[key] == "Not Found" or not details[key]):
                    details[key] = value

            logger.info("Simple fallback successful after enhanced method failed")
        except Exception as simple_e:
            logger.error(f"Simple fallback also failed: {simple_e}")

        return details
async def scrape_amazon_listing_details(url: str) -> dict:
    """
    Scrapes product details from an Amazon listing URL using web search.
    Uses OpenAI's web search capability to extract product information.

    Args:
        url: The Amazon product listing URL

    Returns:
        dict: Product details including title, brand, bullets, description, reviews, and Q&A
    """
    logger.info(f"Attempting to scrape Amazon listing via web search: {url}")

    # Initialize empty product details structure
    details = {
        "title": "Not Found",
        "brand": "Not Found",
        "bullets": [],
        "description": "Not Found",
        "reviews_raw": [],
        "qna_raw": []
    }

    # Check cache first
    if url in CACHE:
        logger.info(f"Using cached data for Amazon URL: {url}")
        return CACHE[url]

    # Extract ASIN from URL if possible
    asin = None
    if "/dp/" in url:
        asin_match = re.search(r'/dp/([A-Z0-9]{10})', url)
        if asin_match:
            asin = asin_match.group(1)

    if not asin and "/product/" in url:
        asin_match = re.search(r'/product/([A-Z0-9]{10})', url)
        if asin_match:
            asin = asin_match.group(1)

    if not asin:
        # Try to find any 10-character alphanumeric string that might be an ASIN
        asin_match = re.search(r'([B][0-9A-Z]{9})', url)
        if asin_match:
            asin = asin_match.group(1)

    logger.info(f"Extracted ASIN: {asin if asin else 'None'}")

    # Prepare for web search
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        # Step 1: Search for basic product information
        basic_info_query = f"{url} product details title brand features description"

        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are a helpful web search assistant."},
                {"role": "user", "content": f"Search for: {basic_info_query}"}
            ],
            "tools": [{"type": "Web Search"}],
            "tool_choice": {"type": "Web Search"}
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )

        basic_info_results = ""
        if response.status_code == 200:
            response_data = response.json()
            tool_calls = response_data["choices"][0]["message"].get("tool_calls", [])
            if tool_calls:
                search_result = tool_calls[0]["function"].get("arguments", "{}")
                basic_info_results = json.loads(search_result).get("search_results", "")
                logger.info("Successfully retrieved basic product information")

        # Step 2: Search for reviews
        reviews_query = f"{url} customer reviews"

        payload["messages"][1]["content"] = f"Search for: {reviews_query}"

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )

        reviews_results = ""
        if response.status_code == 200:
            response_data = response.json()
            tool_calls = response_data["choices"][0]["message"].get("tool_calls", [])
            if tool_calls:
                search_result = tool_calls[0]["function"].get("arguments", "{}")
                reviews_results = json.loads(search_result).get("search_results", "")
                logger.info("Successfully retrieved product reviews")

        # Step 3: Search for Q&A
        qna_query = f"{url} questions and answers FAQ"

        payload["messages"][1]["content"] = f"Search for: {qna_query}"

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )

        qna_results = ""
        if response.status_code == 200:
            response_data = response.json()
            tool_calls = response_data["choices"][0]["message"].get("tool_calls", [])
            if tool_calls:
                search_result = tool_calls[0]["function"].get("arguments", "{}")
                qna_results = json.loads(search_result).get("search_results", "")
                logger.info("Successfully retrieved product Q&A")

        # Step 4: Process all search results with GPT-4 to extract structured information
        combined_search_results = f"""
        BASIC PRODUCT INFORMATION:
        {basic_info_results}

        CUSTOMER REVIEWS:
        {reviews_results}

        QUESTIONS AND ANSWERS:
        {qna_results}
        """

        extraction_prompt = f"""
        Based on the following web search results about an Amazon product, extract structured information in the requested format.

        Amazon Product URL: {url}
        {f"Amazon ASIN: {asin}" if asin else ""}

        WEB SEARCH RESULTS:
        {combined_search_results}

        Extract the following information:
        1. Product Title (full and accurate)
        2. Brand Name
        3. Key Features/Bullet Points (list at least 5 if available)
        4. Product Description (detailed, at least 100 words)
        5. Sample Customer Reviews (3-5 if available)
        6. Common Questions and Answers about the product (2-3 if available)

        For any information you can't find with high confidence, indicate "Not Found" rather than guessing.
        Format your response as a valid JSON object with these keys: title, brand, bullets (array), description, reviews_raw (array), qna_raw (array).

        Ensure the JSON is properly formatted and valid. For reviews_raw and qna_raw, include full text entries, not summaries.
        """

        # Call OpenAI with JSON response format
        extraction_response = call_openai_api(
            prompt=extraction_prompt,
            model=GPT_MODEL,
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        # Parse the response
        extracted_data = json.loads(extraction_response)

        # Update our details with the extracted data
        for key, value in extracted_data.items():
            if key in details:
                details[key] = value

        logger.info("Successfully extracted structured product information from web search results")

        # Cache the results
        CACHE[url] = details
        return details

    except Exception as e:
        logger.error(f"Error during web search-based Amazon scraping for {url}: {e}")

        # Try a simpler fallback approach if the main method fails
        try:
            logger.info(f"Attempting simplified web search fallback for {url}")

            # Create a simpler query from the URL
            product_name = url.split("/")[-1].replace("-", " ").replace(".html", "")
            if asin:
                simple_query = f"Amazon product {asin} {product_name} details"
            else:
                simple_query = f"Amazon product {product_name} details"

            simple_prompt = f"""
            I need information about this Amazon product: {simple_query}
            URL: {url}

            Please provide the following details in a structured format:
            1. Product Title
            2. Brand Name
            3. Key Features (list format)
            4. Product Description
            5. Sample Customer Reviews (if available)
            6. Common Questions and Answers (if available)

            Format as JSON with these keys: title, brand, bullets (array), description, reviews_raw (array), qna_raw (array).
            """

            simple_response = call_openai_api(
                prompt=simple_prompt,
                model=GPT_MODEL,
                temperature=0.3,
                response_format={"type": "json_object"}
            )

            simple_data = json.loads(simple_response)

            # Update our details with the simplified data
            for key, value in simple_data.items():
                if key in details:
                    details[key] = value

            logger.info("Simplified web search fallback successful")

        except Exception as simple_e:
            logger.error(f"Simplified web search fallback also failed: {simple_e}")

        # Cache whatever we have
        CACHE[url] = details
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
        # Using the requests library directly to avoid client issues
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
    
    # Extract reviews and Q&A for deeper analysis
    amazon_reviews = "\n".join(amazon_details.get("reviews_raw", []))
    amazon_qna = "\n".join(amazon_details.get("qna_raw", []))

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

    # Pre-format the JSON structure to avoid nested f-strings
    amazon_listing_value = inputs.get('amazon_listing', '')
    json_structure = (
        '{"editable_sections": [\n'
        '    {"title": "PROJECT OVERVIEW", "questions": [\n'
        '    {"question": "Project Name", "answer": ""},\n'
        '    {"question": "Brand Name", "answer": ""},\n'
        '    {"question": "Website", "answer": ""},\n'
        f'    {{"question": "Amazon Listing (if available)", "answer": "{amazon_listing_value}"}},\n'
        '    {"question": "Instagram Handle (if applicable)", "answer": ""}\n'
        '    ]},\n'
        '    {"title": "PRODUCT SNAPSHOT", "questions": [\n'
        '    {"question": "What exactly is the product?", "answer": ""},\n'
        '    {"question": "What does it do and how does it work?", "answer": ""},\n'
        '    {"question": "What problem does it solve?", "answer": ""},\n'
        '    {"question": "Who is it meant for?", "answer": ""}\n'
        '    ]},\n'
        '    {"title": "CURRENT LISTING CHALLENGES", "questions": [\n'
        '    {"question": "What\'s broken or underwhelming about the current Amazon listing, brand positioning, or creative execution?", "answer": ""},\n'
        '    {"question": "Where are they losing conversions or attention?", "answer": ""}\n'
        '    ]},\n'
        '    {"title": "TARGET CUSTOMER DEEP DIVE", "questions": [\n'
        '    {"question": "Gender, age range, location, income, profession", "answer": ""},\n'
        '    {"question": "Life stage or identity (e.g., new moms, eco-conscious Gen Z, busy professionals)", "answer": ""},\n'
        '    {"question": "Pain points, desires, motivations", "answer": ""},\n'
        '    {"question": "How do they shop on Amazon? What do they care about when scrolling?", "answer": ""}\n'
        '    ]},\n'
        '    {"title": "BARRIERS TO PURCHASE", "questions": [\n'
        '    {"question": "List the common doubts, hesitations, or FAQ-style friction points that stop people from buying — even if they like the product.", "answer": ""}\n'
        '    ]},\n'
        '    {"title": "BRAND VOICE & TONE", "questions": [\n'
        '    {"question": "Describe the tone and copywriting style the brand uses or should use (e.g., bold, sassy, informative, premium, conversational).", "answer": ""},\n'
        '    {"question": "Include any signature words, phrases, or linguistic quirks.", "answer": ""}\n'
        '    ]},\n'
        '    {"title": "USPs (UNIQUE SELLING PROPOSITIONS)", "questions": [\n'
        '    {"question": "What makes this product meaningfully different from other options in the category?", "answer": ""},\n'
        '    {"question": "Think functional benefits, emotional angles, and cultural relevance.", "answer": ""}\n'
        '    ]},\n'
        '    {"title": "5-SECOND WOW FACTOR", "questions": [\n'
        '    {"question": "If a customer saw this listing for 5 seconds, what single visual hook, copy line, or feature would stop them in their tracks?", "answer": ""}\n'
        '    ]},\n'
        '    {"title": "KEY FEATURES (WITH CONTEXT)", "questions": [\n'
        '    {"question": "List 4–6 major features. But go beyond just the bullet points — explain: Why does this matter to the buyer? How does it connect to their lifestyle or values?", "answer": ""}\n'
        '    ]},\n'
        '    {"title": "TOP 6 SELLING POINTS (WITH STRATEGIC JUSTIFICATION)", "questions": [\n'
        '    {"question": "For each of the client\'s selected selling points: State the point. Explain *why* it\'s strategically powerful for this product and customer.", "answer": ""}\n'
        '    ]},\n'
        '    {"title": "COMPETITIVE LANDSCAPE", "questions": [\n'
        '    {"question": "List 2–3 main competitors", "answer": ""},\n'
        '    {"question": "Describe how this product compares", "answer": ""},\n'
        '    {"question": "Mention any Amazon-specific differentiators (e.g. bundle, shipping time, design)", "answer": ""}\n'
        '    ]},\n'
        '    {"title": "SEARCH & KEYWORDS STRATEGY", "questions": [\n'
        '    {"question": "Suggest relevant search terms and niche keywords to target. These should align with user intent, category trends, or long-tail SEO goals.", "answer": ""}\n'
        '    ]},\n'
        '    {"title": "BRAND STORY, VALUES & PURPOSE", "questions": [\n'
        '    {"question": "Give a short but meaningful brand origin story or founder story.", "answer": ""},\n'
        '    {"question": "Highlight core values, emotional drivers, or the \\"bigger why\\" behind the brand\'s existence.", "answer": ""}\n'
        '    ]},\n'
        '    {"title": "DESIGN DIRECTION", "questions": [\n'
        '    {"question": "Summarize the client\'s aesthetic preferences", "answer": ""},\n'
        '    {"question": "Suggest how the visuals, layout, or color themes should feel (e.g., clean/minimal, bold/graphic, warm/natural)", "answer": ""}\n'
        '    ]},\n'
        '    {"title": "FINAL NOTES & STRATEGIC CALLOUTS", "questions": [\n'
        '    {"question": "Include any extra insights for the creative team, such as: Packaging or compliance considerations, Customer education needs, Cross-sell or upsell potential, Social proof or influencer angles", "answer": ""}\n'
        '    ]}\n'
        ']}'
    )

    prompt = f"""
    You are an expert Amazon strategist and copywriter generating a Creative Brief JSON for a high-value client. Your brief will guide the entire creative and marketing strategy for their Amazon product listing.

**PRIMARY DATA SOURCES (In order of priority):**
1. Scraped Amazon Listing Data (Title, Brand, Bullets, Description, Reviews, Q&A)
2. User Provided Input (Brand details, product category, challenges)
3. Scraped Website Content (supplementary context only)
4. Web search data (if scraping methods failed)

**MASTER CREATIVE BRIEF FRAMEWORK:**

**CRITICAL INSTRUCTIONS:**
1. Generate a **comprehensive, strategic Creative Brief** in the specified JSON format that will serve as the foundation for a complete Amazon listing overhaul.
2. Answer **ALL questions** with detailed, product-specific content that demonstrates deep market understanding.
3. Derive answers **PRIMARILY** from the "Scraped Amazon Listing Data" - especially customer reviews and Q&A which contain critical VOC (Voice of Customer) insights.
4. Use "User Provided Input" to confirm information or fill gaps not found on Amazon.
5. Use "Scraped Website Content" only when information is missing from both Amazon and User Input.
6. **NEVER LEAVE ANSWERS BLANK OR GENERIC.** If information isn't explicitly stated, make strategic inferences based on:
   - Product category norms and best practices
   - Competitive landscape analysis
   - Customer language patterns in reviews/Q&A
   - Amazon marketplace trends
   
   Always clearly label inferences as "Strategic recommendation based on [specific insight]" so the client knows what's data-driven vs. strategic guidance.

7. **FOCUS ON ACTIONABLE INSIGHTS** that directly impact conversion rate and discoverability:
   - Use specific customer language from reviews in your recommendations
   - Identify precise pain points that block purchase
   - Suggest concrete messaging approaches, not vague platitudes
   - Recommend specific visual elements that would enhance the listing

8. **MAINTAIN STRATEGIC CONSISTENCY** across all sections - ensure your target audience, tone, USPs, and selling points align with each other.

9. **IDENTIFY WEB SEARCH DATA** in the "FINAL NOTES & STRATEGIC CALLOUTS" section with:
   "Note: Some product information was obtained through AI-assisted web search due to scraping limitations. The following sections contain data from web search that should be verified with the client: [List specific sections]"

**ADVANCED STRATEGIC ANALYSIS FRAMEWORK:**

**For PRODUCT SNAPSHOT:**
- Define the product with precision, focusing on its core function and category placement
- Explain its operation using technical details from bullets/description, emphasizing unique mechanisms
- Identify the primary customer problem by analyzing negative competitor reviews and "pain point" language in positive reviews
- Define the target audience using demographic and psychographic indicators from review language patterns

**For CURRENT LISTING CHALLENGES:**
- Analyze the current listing for specific conversion barriers:
  * Missing information (common questions in Q&A indicate gaps)
  * Unclear value proposition (reviews expressing surprise at features suggest poor communication)
  * Weak imagery (comments about "looks different than expected")
  * Price justification issues (reviews mentioning value or comparisons)
  * Trust barriers (reviews mentioning hesitation before purchase)
- Identify specific attention-losing elements by comparing to category best practices

**For TARGET CUSTOMER DEEP DIVE:**
- Extract demographic details from review language patterns:
  * Gender identity markers in language
  * Age indicators (references to life stage, technology comfort, etc.)
  * Income signals (price sensitivity comments, luxury expectations)
  * Professional context mentions
- Identify psychographic segments based on:
  * Value expressions in reviews ("I care about sustainability")
  * Lifestyle references ("perfect for my morning routine")
  * Purchase motivation patterns ("needed something for...")
- Map Amazon shopping behavior based on:
  * Decision factors mentioned in reviews
  * Questions asked before purchase
  * Comparison shopping references

**For BARRIERS TO PURCHASE:**
- Extract specific doubts from:
  * Questions in Q&A section (especially repeated questions)
  * "Almost didn't buy because..." statements in reviews
  * Negative reviews of competitors in same category
  * Hesitations mentioned in positive reviews ("was worried about X but...")
- Categorize barriers as:
  * Product understanding gaps
  * Quality/durability concerns
  * Compatibility/fit uncertainties
  * Value justification needs
  * Social proof requirements

**For BRAND VOICE & TONE:**
- Analyze existing brand language across:
  * Amazon bullet structure and syntax patterns
  * Description paragraph style and sentence construction
  * Website messaging approach
  * Social media voice if available
- Identify specific language patterns:
  * Sentence length and complexity
  * Technical vs. conversational balance
  * Emotional vs. rational appeals
  * Use of humor, questions, commands
  * Formality level and jargon density
- Recommend specific voice characteristics with examples

**For USPs (UNIQUE SELLING PROPOSITIONS):**
- Identify true differentiators by analyzing:
  * Features mentioned most positively in reviews
  * Comparison statements to other products
  * "The reason I chose this..." statements
  * Repeated praise patterns across multiple reviews
- Structure USPs as problem/solution pairs
- Validate USPs against competitor offerings
- Prioritize USPs based on review frequency and emotional intensity

**For 5-SECOND WOW FACTOR:**
- Extract the most emotionally impactful feature from:
  * "I was amazed by..." statements in reviews
  * Exclamation-heavy comments
  * Features that solved long-standing customer problems
  * Unexpected benefits mentioned with surprise
- Craft a concise, high-impact statement that combines:
  * The key differentiator
  * The emotional benefit
  * A visual element when possible

**For KEY FEATURES (WITH CONTEXT):**
- Extract 4-6 most substantive features from:
  * Amazon bullets (focusing on specifications, materials, dimensions)
  * Technical details in description
  * Features mentioned positively in multiple reviews
- For each feature, connect to:
  * A specific customer problem it solves
  * A lifestyle integration point
  * An emotional benefit
- Format as feature → context → benefit chains

**For TOP 6 SELLING POINTS:**
- Identify the strongest conversion drivers by analyzing:
  * Most frequently mentioned positive attributes in reviews
  * Features that drove purchase decisions
  * Benefits that exceeded customer expectations
  * Points of differentiation from competitors
- For each selling point, provide:
  * The concrete claim/feature
  * Evidence from customer feedback
  * Strategic relevance to target audience
  * Competitive advantage context

**For COMPETITIVE LANDSCAPE:**
- Identify direct competitors through:
  * Brand comparisons in reviews ("better than X")
  * "Also considered..." statements
  * Same-category bestsellers
- Compare on specific dimensions:
  * Price positioning
  * Feature set overlap and gaps
  * Quality perception
  * Target audience alignment
- Highlight Amazon-specific advantages:
  * Prime eligibility impact
  * Bundle opportunities
  * Review volume/rating advantages
  * Search ranking potential

**For SEARCH & KEYWORDS STRATEGY:**
- Extract high-conversion search terms from:
  * Product category terminology in reviews
  * Problem statements in customer language
  * Specific feature descriptions in customer words
  * Use case scenarios mentioned in reviews
- Identify long-tail opportunities based on:
  * Specific questions in Q&A
  * Niche use cases mentioned in reviews
  * Problem-specific language patterns
- Organize keywords by:
  * Purchase intent level
  * Specificity to product
  * Competition level
  * Relevance to target audience

**For BRAND STORY, VALUES & PURPOSE:**
- Construct a compelling narrative using:
  * Origin elements from website/about page
  * Value statements in product description
  * Mission-focused language in marketing materials
  * Customer perception of brand in reviews
- Structure as:
  * Founding problem/insight
  * Brand solution approach
  * Customer-centered promise
  * Emotional connection point
- Identify core values based on:
  * Repeated themes in brand communication
  * Customer appreciation points in reviews
  * Differentiation elements from competitors

**For DESIGN DIRECTION:**
- Analyze current visual approach for:
  * Color palette consistency
  * Typography style and hierarchy
  * Image composition patterns
  * Graphic element usage
- Recommend specific improvements to:
  * Hero image composition
  * Infographic clarity
  * Lifestyle image authenticity
  * Mobile optimization
  * A+ content structure
- Provide specific visual do's and don'ts

**For FINAL NOTES & STRATEGIC CALLOUTS:**
- Highlight critical implementation considerations:
  * Compliance requirements specific to category
  * Technical limitations of Amazon platform
  * Seasonal/timing strategic opportunities
  * Cross-sell/bundle potential with specific products
  * Review generation strategy recommendations
- Note any data sources that require verification
- Suggest next steps for implementation prioritization

---
### Scraped Amazon Listing Data (PRIMARY SOURCE):
**Brand:** {amazon_brand}
**Product Title:** {amazon_title}
**Bullet Points:**
{amazon_bullets_str}
**Product Description:**
{amazon_description[:2500]}...
**Customer Reviews:**
{amazon_reviews[:1500]}...
**Customer Q&A:**
{amazon_qna[:1500]}...
**Customer Insights (VOC Analysis):**
{voc_insights}
---
### User Provided Input (Secondary Source):
{user_input_summary}
{guideline_section}
---
{website_section}
---

**JSON Structure to Generate:**
{json_structure}

Your task is to generate a comprehensive, strategic Creative Brief that will transform this Amazon listing. Focus on extracting deep customer insights from reviews and Q&A, identifying true differentiators, and providing actionable guidance that directly impacts conversion. Every section should contain specific, detailed recommendations based on data when available or strategic expertise when data is limited.

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
        # For Render deployment, the service account file should be at this location
        service_account_path = "/etc/secrets/service_account.json"

        if not os.path.exists(service_account_path):
            logger.error(f"Service account file not found at {service_account_path}")
            # Try fallback location
            service_account_path = os.getenv("SERVICE_ACCOUNT_FILE", "/opt/render/project/src/service_account.json")
            if not os.path.exists(service_account_path):
                logger.error(f"Service account file not found at fallback location {service_account_path}")
                return None, None

        logger.info(f"Using service account file at {service_account_path}")

        # Debug file content
        try:
            with open(service_account_path, 'r') as f:
                content = f.read()
                logger.info(f"Service account file size: {len(content)} bytes")
                if not content.strip():
                    logger.error("Service account file is empty")
                    return None, None

                # Try parsing the JSON to see if it's valid
                try:
                    json_content = json.loads(content)
                    logger.info("Service account JSON is valid")
                except json.JSONDecodeError as json_err:
                    logger.error(f"Invalid JSON in service account file: {str(json_err)}")
                    return None, None
        except Exception as file_err:
            logger.error(f"Error reading service account file: {str(file_err)}")
            return None, None

        # Continue with normal flow
        creds = service_account.Credentials.from_service_account_file(
            service_account_path, scopes=SCOPES)

        service = build('drive', 'v3', credentials=creds)

        # Get the mime type based on file extension
        mime_type = 'application/pdf'  # Use PDF mime type

        # Use the correct folder ID directly
        file_metadata = {
            'name': filename,
            'parents': [GOOGLE_DRIVE_FOLDER_ID]  # Add to specified folder
        }

        media = MediaFileUpload(file_path, mimetype=mime_type)
        file = service.files().create(body=file_metadata, media_body=media, fields='id, webViewLink').execute()

        logger.info(f'File ID: {file.get("id")}')
        logger.info(f'File Link: {file.get("webViewLink")}')

        return file.get('id'), file.get('webViewLink')
    except Exception as e:
        logger.error(f"Error uploading to Google Drive: {str(e)}")
        return None, None

# --- FASTAPI ROUTES ---
@app.api_route("/", methods=["GET", "HEAD"])
async def root(request: Request):
    if request.method == "HEAD":
        # For HEAD requests, return an empty response with 200 status
        return {}
    # For GET requests, return your normal response
    return {"status": "online", "message": "Creative Brief Generator API is running", "docs_url": "/docs"}

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