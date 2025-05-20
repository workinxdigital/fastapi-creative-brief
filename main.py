import os
import json
import re
import shutil
import logging
from io import BytesIO
from typing import List, Optional, Union, Dict, Any
import asyncio
import time

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
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, PageBreak
from reportlab.pdfgen.canvas import Canvas
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

# PDF Heading Constants - Add these lines
# PDF Heading Constants
MAIN_HEADINGS = {
    "TITLE": "Creative Brief",
    "ASSETS_OVERVIEW": "ASSETS OVERVIEW",
    "PROJECT_OVERVIEW": "PROJECT OVERVIEW",
    "PRODUCT_SNAPSHOT": "PRODUCT SNAPSHOT",
    "CURRENT_LISTING_CHALLENGES": "CURRENT LISTING CHALLENGES",
    "TARGET_CUSTOMER": "TARGET CUSTOMER DEEP DIVE",
    "BARRIERS_TO_PURCHASE": "BARRIERS TO PURCHASE",
    "BRAND_VOICE": "BRAND VOICE & TONE",
    "USPS": "USPs (UNIQUE SELLING PROPOSITIONS)",
    "WOW_FACTOR": "5-SECOND WOW FACTOR",
    "KEY_FEATURES": "KEY FEATURES (WITH CONTEXT)",
    "TOP_SELLING_POINTS": "TOP 6 SELLING POINTS (WITH STRATEGIC JUSTIFICATION)",
    "COMPETITIVE_LANDSCAPE": "COMPETITIVE LANDSCAPE",
    "SEARCH_KEYWORDS": "SEARCH & KEYWORDS STRATEGY",
    "BRAND_STORY": "BRAND STORY, VALUES & PURPOSE",
    "DESIGN_DIRECTION": "DESIGN DIRECTION",
    "FINAL_NOTES": "FINAL NOTES & STRATEGIC CALLOUTS"
}

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

# --- UNIFIED WEB SEARCH SCRAPING ENGINE ---
class WebSearchScraper:
    """
    Unified Web Search Scraping Engine that handles all web search operations
    with configurable search strategies for different content types.
    """
    
    def __init__(self, api_key: str = OPENAI_API_KEY, model: str = "gpt-4o", cache_enabled: bool = True):
        self.api_key = api_key
        self.model = model
        self.cache = {}
        self.cache_enabled = cache_enabled
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
    def _get_cache_key(self, query_type: str, url: str = "", query: str = "") -> str:
        """Generate a unique cache key based on query type and parameters"""
        if url:
            return f"{query_type}:{url}"
        return f"{query_type}:{query}"
        
    def _check_cache(self, cache_key: str) -> Optional[Dict]:
        """Check if result exists in cache"""
        if not self.cache_enabled:
            return None
        if cache_key in self.cache:
            logger.info(f"Cache hit for {cache_key}")
            return self.cache[cache_key]
        return None
        
    def _update_cache(self, cache_key: str, result: Dict) -> None:
        """Update cache with new result"""
        if self.cache_enabled:
            self.cache[cache_key] = result
            logger.info(f"Updated cache for {cache_key}")
    
    async def search(self, query: str, num_results: int = 5, search_type: str = "general") -> List[Dict]:
        """
        Perform a web search with configurable parameters
        
        Args:
            query: The search query
            num_results: Number of results to return
            search_type: Type of search (general, product, reviews, etc.)
            
        Returns:
            List of search result dictionaries
        """
        cache_key = self._get_cache_key(f"search:{search_type}", query=query)
        cached_result = self._check_cache(cache_key)
        if cached_result:
            return cached_result
            
        try:
            # Configure search based on type
            system_message = "You are a helpful web search assistant."
            if search_type == "product":
                system_message = "You are a helpful web search assistant focused on finding detailed product information."
            elif search_type == "reviews":
                system_message = "You are a helpful web search assistant focused on finding customer reviews and feedback."
            elif search_type == "brand":
                system_message = "You are a helpful web search assistant focused on finding brand information and company details."
                
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"Search for: {query}"}
                ],
                "tools": [{"type": "Web Search"}],
                "tool_choice": {"type": "Web Search"}
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=self.headers,
                json=payload
            )
            
            if response.status_code != 200:
                logger.error(f"Web search API error: {response.status_code} - {response.text}")
                return []
                
            response_data = response.json()
            tool_calls = response_data["choices"][0]["message"].get("tool_calls", [])
            
            if not tool_calls:
                logger.warning("No tool calls found in web search response")
                return []
                
            search_results = json.loads(tool_calls[0]["function"].get("arguments", "{}"))
            results = search_results.get("search_results", [])
            
            # Process and structure the results
            processed_results = []
            for result in results[:num_results]:
                processed_results.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "snippet": result.get("snippet", "")
                })
                
            self._update_cache(cache_key, processed_results)
            return processed_results
            
        except Exception as e:
            logger.error(f"Error in web search: {e}")
            return []
    
    # Remove the Playwright import
# from playwright.async_api import async_playwright

# Update the WebSearchScraper.scrape_website method to use web search instead of Playwright
async def scrape_website(self, url: str) -> str:
    """
    Scrape text content from a website using web search

    Args:
        url: The website URL to scrape

    Returns:
        Extracted text content from the website
    """
    cache_key = self._get_cache_key("website", url=url)
    cached_result = self._check_cache(cache_key)
    if cached_result:
        return cached_result

    logger.info(f"Attempting to scrape website via web search: {url}")
    try:
        # Extract domain and path for better search
        domain = url.split("//")[-1].split("/")[0]

        # Extract hints from URL path
        path_parts = url.split("/")
        content_hint = ""
        if len(path_parts) > 3:
            content_hint = path_parts[-1].replace("-", " ").replace(".html", "").replace(".php", "")

        query = f"{domain} {content_hint} website content information"
        search_results = await self.search(query, num_results=5, search_type="general")

        if not search_results:
            return f"Could not retrieve content from {url}"

        # Combine search results into a text representation
        combined_content = f"Website: {url}\n\n"
        for result in search_results:
            if domain in result.get("url", ""):
                combined_content += f"Title: {result.get('title', '')}\n"
                combined_content += f"Content: {result.get('snippet', '')}\n\n"

        self._update_cache(cache_key, combined_content)
        return combined_content

    except Exception as e:
        logger.error(f"Error scraping website via web search: {e}")
        return f"Could not scrape website: {e}"
    
    async def extract_amazon_listing(self, url: str, include_storefront: bool = False) -> Dict[str, Any]:
        """
        Extract comprehensive information from an Amazon product listing
        
        Args:
            url: The Amazon product listing URL
            include_storefront: Whether to also extract storefront information
            
        Returns:
            Dictionary containing structured product information
        """
        cache_key = self._get_cache_key("amazon", url=url)
        cached_result = self._check_cache(cache_key)
        if cached_result:
            return cached_result
            
        logger.info(f"Extracting Amazon listing details for: {url}")
        
        # Initialize empty product details structure
        details = {
            "title": "Not Found",
            "brand": "Not Found",
            "bullets": [],
            "description": "Not Found",
            "reviews_raw": [],
            "qna_raw": [],
            "technical_details": {},
            "enhanced_content": "",
            "storefront_data": None
        }
        
        # Extract ASIN and product name from URL for better search
        asin = None
        product_name = ""
        brand_name = ""
        
        # Extract ASIN using regex patterns
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
        
        # Extract product name from URL
        name_parts = url.split('/')
        for part in name_parts:
            if part and '-' in part and len(part) > 10 and not part.startswith('dp') and not part.startswith('B0'):
                product_name = part.replace('-', ' ')
                break
        
        # Extract potential brand name
        if "brand=" in url:
            brand_parts = url.lower().split("brand=")
            if len(brand_parts) > 1:
                brand_name = brand_parts[1].split("&")[0].replace("+", " ")
        
        logger.info(f"Extracted ASIN: {asin if asin else 'None'}")
        logger.info(f"Extracted product name: {product_name}")
        logger.info(f"Extracted potential brand: {brand_name}")
        
        try:
            search_results = []
            
            # Step 1: Direct search for the exact Amazon product page
            exact_product_query = f"{url} amazon product page"
            product_results = await self.search(exact_product_query, num_results=5, search_type="product")
            if product_results:
                search_results.extend(product_results)
            
            # Step 2: Search for product specifications and features
            specs_query = f"amazon {asin if asin else product_name} product specifications features details"
            specs_results = await self.search(specs_query, num_results=5, search_type="product")
            if specs_results:
                search_results.extend(specs_results)
            
            # Step 3: Search for product reviews
            reviews_query = f"amazon {asin if asin else product_name} customer reviews ratings"
            reviews_results = await self.search(reviews_query, num_results=5, search_type="reviews")
            if reviews_results:
                search_results.extend(reviews_results)
            
            # Step 4: Search for product Q&A
            qna_query = f"amazon {asin if asin else product_name} questions and answers FAQ"
            qna_results = await self.search(qna_query, num_results=5, search_type="general")
            if qna_results:
                search_results.extend(qna_results)
            
            # Step 5: Search for brand information
            if brand_name or "brand" in url.lower() or product_name:
                # Try to extract brand from URL or product name
                potential_brand = brand_name
                
                if not potential_brand and product_name:
                    # Take first 1-2 words as potential brand
                    name_parts = product_name.split()
                    if len(name_parts) > 0:
                        potential_brand = " ".join(name_parts[:min(2, len(name_parts))])
                
                if potential_brand:
                    brand_query = f"{potential_brand} brand information amazon"
                    brand_results = await self.search(brand_query, num_results=5, search_type="brand")
                    if brand_results:
                        search_results.extend(brand_results)
            
            # Step 6: If storefront extraction is requested, search for brand storefront
            storefront_data = None
            if include_storefront and (brand_name or potential_brand):
                brand_to_search = brand_name if brand_name else potential_brand
                storefront_query = f"amazon {brand_to_search} brand store storefront page"
                storefront_results = await self.search(storefront_query, num_results=5, search_type="brand")
                
                if storefront_results:
                    # Process storefront results
                    storefront_data = await self._extract_storefront_data(brand_to_search, storefront_results)
                    details["storefront_data"] = storefront_data
            
            # Step 7: Process all search results with GPT-4 to extract structured information
            combined_search_results = ""
            for result in search_results:
                combined_search_results += f"Title: {result.get('title', '')}\n"
                combined_search_results += f"URL: {result.get('url', '')}\n"
                combined_search_results += f"Content: {result.get('snippet', '')}\n\n"
            
            extraction_prompt = f"""
            You are an expert Amazon product data extractor. Your task is to extract detailed, accurate product information from web search results about an Amazon product.

            Amazon Product URL: {url}
            {f"Amazon ASIN: {asin}" if asin else ""}
            {f"Product Name: {product_name}" if product_name else ""}
            {f"Brand Name: {brand_name}" if brand_name else ""}

            WEB SEARCH RESULTS:
            {combined_search_results}

            Extract the following information with high precision and detail:

            1. Product Title (full and accurate, including brand and model number if available)
            2. Brand Name (just the brand name without additional text like "by" or "from")
            3. Key Features/Bullet Points (extract at least 5-8 if available, focusing on technical specifications, materials, dimensions, and key selling points)
            4. Product Description (provide a detailed, comprehensive description that covers what the product is, how it works, and its benefits)
            5. Technical Details (extract technical specifications in a structured format, including dimensions, weight, materials, compatibility, etc.)
            6. Enhanced Content (extract any A+ content, comparison charts, or brand story sections)
            7. Customer Reviews (extract 5-10 detailed reviews that provide specific feedback about the product, including both positive and negative perspectives)
            8. Questions and Answers (extract 3-5 detailed Q&A pairs that provide insights about common customer concerns or usage information)
            9. Competitor Comparison (extract any information comparing this product to alternatives)
            10. Seller Information (extract details about the seller and fulfillment)

            IMPORTANT EXTRACTION GUIDELINES:
            - For the title, extract the EXACT product title as it appears on Amazon, including all model numbers and specifications
            - For bullet points, maintain the original formatting and technical details, avoiding generic marketing language
            - For the description, provide a comprehensive overview that includes technical details, use cases, and benefits
            - For technical details, organize them into a structured format with clear labels
            - For enhanced content, capture any special product presentations, comparison charts, or brand storytelling
            - For reviews, include the reviewer's rating if available (e.g., "5-star review:") and focus on detailed, specific feedback
            - For Q&A, include complete questions and their corresponding answers
            - If information is not available or unclear, indicate "Not Found" rather than guessing
            - Ensure all extracted information is specific to THIS product, not similar or related products

            Format your response as a valid JSON object with these keys: 
            title, brand, bullets (array), description, technical_details (object), enhanced_content, reviews_raw (array), qna_raw (array), competitor_comparison, seller_info.
            """
            
            # Call OpenAI with JSON response format
            extraction_response = await self._call_openai_api(
                prompt=extraction_prompt,
                model=GPT_MODEL,
                temperature=0.2,  # Lower temperature for more factual extraction
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            extracted_data = json.loads(extraction_response)
            
            # Update details with extracted data
            for key in extracted_data:
                if key in details and extracted_data[key]:
                    details[key] = extracted_data[key]
            
            # Add additional fields if they exist in extracted data
            if "technical_details" in extracted_data and extracted_data["technical_details"]:
                details["technical_details"] = extracted_data["technical_details"]
            
            if "enhanced_content" in extracted_data and extracted_data["enhanced_content"]:
                details["enhanced_content"] = extracted_data["enhanced_content"]
                
            if "competitor_comparison" in extracted_data and extracted_data["competitor_comparison"]:
                details["competitor_comparison"] = extracted_data["competitor_comparison"]
                
            if "seller_info" in extracted_data and extracted_data["seller_info"]:
                details["seller_info"] = extracted_data["seller_info"]
            
            # Clean and validate the data
            details = self._clean_amazon_data(details)
            
            # Cache the results
            self._update_cache(cache_key, details)
            return details
            
        except Exception as e:
            logger.error(f"Error extracting Amazon listing details: {e}")
            
            # Try a simpler fallback approach
            try:
                logger.info(f"Attempting simplified extraction fallback for {url}")
                simple_query = f"Amazon product {asin if asin else product_name} complete details"
                simple_results = await self.search(simple_query, num_results=8, search_type="product")
                
                if not simple_results:
                    return details
                
                # Combine simple results
                simple_combined = ""
                for result in simple_results:
                    simple_combined += f"{result.get('title', '')}\n{result.get('snippet', '')}\n\n"
                
                simple_prompt = f"""
                Extract comprehensive information about this Amazon product:
                URL: {url}
                {f"ASIN: {asin}" if asin else ""}
                {f"Product Name: {product_name}" if product_name else ""}

                SEARCH RESULTS:
                {simple_combined}

                Extract and format as JSON with these keys: 
                title, brand, bullets (array), description, technical_details (object), enhanced_content, reviews_raw (array), qna_raw (array).
                """
                
                simple_response = await self._call_openai_api(
                    prompt=simple_prompt,
                    model=GPT_MODEL,
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )
                
                simple_data = json.loads(simple_response)
                
                # Update our details with the simplified data
                for key, value in simple_data.items():
                    if key in details and value:
                        details[key] = value
                
                # Clean and validate the data
                details = self._clean_amazon_data(details)
                
                # Cache the results
                self._update_cache(cache_key, details)
                
            except Exception as simple_e:
                logger.error(f"Simplified extraction fallback also failed: {simple_e}")
            
            return details
    
    async def _extract_storefront_data(self, brand_name: str, storefront_results: List[Dict]) -> Dict[str, Any]:
        """
        Extract Amazon storefront information for a brand
        
        Args:
            brand_name: The brand name to extract storefront data for
            storefront_results: Search results related to the brand storefront
            
        Returns:
            Dictionary containing structured storefront information
        """
        logger.info(f"Extracting storefront data for brand: {brand_name}")
        
        # Combine storefront search results
        combined_storefront = ""
        storefront_url = None
        
        for result in storefront_results:
            result_url = result.get('url', '')
            if "amazon.com/stores" in result_url or "amazon.com/shop" in result_url:
                storefront_url = result_url
            combined_storefront += f"Title: {result.get('title', '')}\n"
            combined_storefront += f"URL: {result_url}\n"
            combined_storefront += f"Content: {result.get('snippet', '')}\n\n"
        
        storefront_prompt = f"""
        You are an expert Amazon brand storefront analyst. Extract comprehensive information about this brand's Amazon storefront.

        Brand Name: {brand_name}
        Storefront URL: {storefront_url if storefront_url else "Not explicitly found in search results"}

        SEARCH RESULTS:
        {combined_storefront}

        Extract the following information about the brand's Amazon storefront:

        1. Brand Overview: A comprehensive summary of what the brand is about, its positioning, and its market focus
        2. Brand Mission/Values: Any stated mission, values, or commitments the brand makes
        3. Featured Products: Key product categories or flagship products highlighted in the storefront
        4. Brand Story: Any narrative about the brand's history, founding, or journey
        5. Visual Elements: Description of the brand's visual identity, imagery themes, and aesthetic
        6. Special Promotions: Any current promotions, deals, or special offerings
        7. Customer Testimonials: Any highlighted customer feedback or testimonials specific to the brand
        8. Brand Differentiators: What makes this brand stand out from competitors based on their storefront

        If certain information isn't available in the search results, indicate "Not Found" for that section.
        Format your response as a valid JSON object with these keys: 
        overview, mission_values, featured_products (array), brand_story, visual_elements, promotions, testimonials (array), differentiators (array).
        """
        
        try:
            storefront_response = await self._call_openai_api(
                prompt=storefront_prompt,
                model=GPT_MODEL,
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            storefront_data = json.loads(storefront_response)
            
            # Add the storefront URL if found
            if storefront_url:
                storefront_data["storefront_url"] = storefront_url
                
            return storefront_data
            
        except Exception as e:
            logger.error(f"Error extracting storefront data: {e}")
            return {
                "overview": "Not Found",
                "mission_values": "Not Found",
                "featured_products": [],
                "brand_story": "Not Found",
                "visual_elements": "Not Found",
                "promotions": "Not Found",
                "testimonials": [],
                "differentiators": []
            }
    
    def _clean_amazon_data(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean and validate Amazon product data
        
        Args:
            details: Raw extracted product details
            
        Returns:
            Cleaned and validated product details
        """
        # Clean bullet points - remove duplicates and empty entries
        if "bullets" in details and isinstance(details["bullets"], list):
            cleaned_bullets = []
            seen_bullets = set()
            for bullet in details["bullets"]:
                # Normalize bullet text for deduplication
                normalized = re.sub(r'\s+', ' ', str(bullet).lower().strip())
                if normalized and normalized not in seen_bullets and len(normalized) > 5:
                    cleaned_bullets.append(bullet)
                    seen_bullets.add(normalized)
            details["bullets"] = cleaned_bullets
        
        # Clean reviews - remove duplicates and ensure they're substantial
        if "reviews_raw" in details and isinstance(details["reviews_raw"], list):
            cleaned_reviews = []
            seen_reviews = set()
            for review in details["reviews_raw"]:
                # Normalize review text for deduplication
                normalized = re.sub(r'\s+', ' ', str(review).lower().strip())
                if normalized and normalized not in seen_reviews and len(normalized) > 20:
                    cleaned_reviews.append(review)
                    seen_reviews.add(normalized)
            details["reviews_raw"] = cleaned_reviews
        
        # Clean Q&A - ensure they're in Q&A format and substantial
        if "qna_raw" in details and isinstance(details["qna_raw"], list):
            cleaned_qna = []
            for qna in details["qna_raw"]:
                if ("Q:" in str(qna) or "Question:" in str(qna)) and ("A:" in str(qna) or "Answer:" in str(qna)) and len(str(qna)) > 20:
                    cleaned_qna.append(qna)
                elif "?" in str(qna) and len(str(qna)) > 20:
                    # Try to format as Q&A if it contains a question mark
                    parts = str(qna).split("?", 1)
                    if len(parts) == 2:
                        formatted_qna = f"Q: {parts[0]}? A: {parts[1].strip()}"
                        cleaned_qna.append(formatted_qna)
            details["qna_raw"] = cleaned_qna
            
        return details
    
    async def _call_openai_api(self, prompt: str, model: str = GPT_MODEL, temperature: float = 0.5, response_format: dict = None) -> str:
        """
        Call OpenAI API with retry logic
        
        Args:
            prompt: The prompt to send to OpenAI
            model: The model to use
            temperature: Temperature setting for generation
            response_format: Optional response format specification
            
        Returns:
            The API response content
        """
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "You are an expert data extraction assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": temperature,
                    "max_tokens": 4096
                }
                
                if response_format:
                    payload["response_format"] = response_format
                
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=self.headers,
                    json=payload
                )
                
                if response.status_code != 200:
                    logger.error(f"OpenAI API returned status code {response.status_code}: {response.text}")
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        logger.info(f"Retrying in {wait_time} seconds...")
                        await asyncio.sleep(wait_time)
                        continue
                    raise Exception(f"OpenAI API error: {response.text}")
                
                response_data = response.json()
                content = response_data["choices"][0]["message"]["content"]
                return content
                
            except Exception as e:
                logger.error(f"OpenAI API error (attempt {attempt+1}/{max_retries}): {e}")
                if "token" in str(e).lower() and len(prompt) > 4000:
                    logger.info("Token limit likely exceeded, retrying with shorter prompt.")
                    prompt = prompt[:4000] + "\n[Content truncated due to length. Please summarize based on available data.]"
                
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    raise Exception(f"Failed to call OpenAI API after {max_retries} attempts: {e}")

# --- SCRAPING FUNCTIONS ---
async def scrape_website_text(url: str) -> str:
    """
    Scrape text content from a website
    
    Args:
        url: The website URL to scrape
        
    Returns:
        Extracted text content from the website
    """
    scraper = WebSearchScraper()
    return await scraper.scrape_website(url)

async def scrape_amazon_listing_details(url: str, include_storefront: bool = False) -> dict:
    """
    Scrapes product details from an Amazon listing URL using web search.
    Uses OpenAI's web search capability to extract comprehensive product information.

    Args:
    url: The Amazon product listing URL
    include_storefront: Whether to also extract storefront information

    Returns:
    dict: Product details including title, brand, bullets, description, reviews, Q&A, and storefront data if requested
    """
    logger.info(f"Attempting to scrape Amazon listing via web search: {url}")

    # Initialize empty product details structure
    details = {
        "title": "Not Found",
        "brand": "Not Found",
        "bullets": [],
        "description": "Not Found",
        "reviews_raw": [],
        "qna_raw": [],
        "storefront_data": None  # Add storefront data field
    }

    # Check cache first
    cache_key = f"amazon:{url}"
    if cache_key in CACHE:
        logger.info(f"Using cached data for Amazon URL: {url}")
        return CACHE[cache_key]

    # Extract ASIN and product name from URL for better search
    asin = None
    product_name = ""
    brand_name = ""

    # Extract ASIN using regex patterns
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

    # Extract product name from URL
    name_parts = url.split('/')
    for part in name_parts:
        if part and '-' in part and len(part) > 10 and not part.startswith('dp') and not part.startswith('B0'):
            product_name = part.replace('-', ' ')
            break

    # Extract potential brand name
    if "brand=" in url:
        brand_parts = url.lower().split("brand=")
        if len(brand_parts) > 1:
            brand_name = brand_parts[1].split("&")[0].replace("+", " ")

    logger.info(f"Extracted ASIN: {asin if asin else 'None'}")
    logger.info(f"Extracted product name: {product_name}")
    logger.info(f"Extracted potential brand: {brand_name if brand_name else product_name.split()[0] if product_name else ''}")

    # Prepare for web search
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        search_results = []

        # Step 1: Direct search for the exact Amazon product page
        exact_product_query = f"{url} amazon product page"

        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are a helpful web search assistant focused on finding accurate product information."},
                {"role": "user", "content": f"Search for: {exact_product_query}"}
            ],
            "tools": [{
                "type": "Web Search",
                "function": {
                    "name": "Web Search",
                    "parameters": {}
                }
            }],
            "tool_choice": {"type": "function", "function": {"name": "Web Search"}}
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
                search_result = tool_calls[0]["function"].get("arguments", "{}")
                search_results.append(json.loads(search_result).get("search_results", ""))
                logger.info("Successfully retrieved direct product page information")

        # Step 2: Search for product specifications and features
        specs_query = f"amazon {asin if asin else product_name} product specifications features details"

        payload["messages"][1]["content"] = f"Search for: {specs_query}"

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )

        if response.status_code == 200:
            response_data = response.json()
            tool_calls = response_data["choices"][0]["message"].get("tool_calls", [])
            if tool_calls:
                search_result = tool_calls[0]["function"].get("arguments", "{}")
                search_results.append(json.loads(search_result).get("search_results", ""))
                logger.info("Successfully retrieved product specifications")

        # Step 3: Search for product reviews
        reviews_query = f"amazon {asin if asin else product_name} customer reviews ratings"

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
                search_result = tool_calls[0]["function"].get("arguments", "{}")
                search_results.append(json.loads(search_result).get("search_results", ""))
                logger.info("Successfully retrieved product reviews")

        # Step 4: Search for product Q&A
        qna_query = f"amazon {asin if asin else product_name} questions and answers FAQ"

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
                search_result = tool_calls[0]["function"].get("arguments", "{}")
                search_results.append(json.loads(search_result).get("search_results", ""))
                logger.info("Successfully retrieved product Q&A")

        # Step 5: Search for brand information
        potential_brand = ""
        if brand_name or "brand" in url.lower() or product_name:
            # Try to extract brand from URL or product name
            potential_brand = brand_name

            if not potential_brand and "brand" in url.lower():
                brand_parts = url.lower().split("brand=")
                if len(brand_parts) > 1:
                    potential_brand = brand_parts[1].split("&")[0].replace("+", " ")

            if not potential_brand and product_name:
                # Take first 1-2 words as potential brand
                name_parts = product_name.split()
                if len(name_parts) > 0:
                    potential_brand = " ".join(name_parts[:min(2, len(name_parts))])

            if potential_brand:
                brand_query = f"{potential_brand} brand information amazon"

                payload["messages"][1]["content"] = f"Search for: {brand_query}"

                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload
                )

                if response.status_code == 200:
                    response_data = response.json()
                    tool_calls = response_data["choices"][0]["message"].get("tool_calls", [])
                    if tool_calls:
                        search_result = tool_calls[0]["function"].get("arguments", "{}")
                        search_results.append(json.loads(search_result).get("search_results", ""))
                        logger.info(f"Successfully retrieved brand information for {potential_brand}")

        # Step 6: If storefront extraction is requested, search for brand storefront
        storefront_data = None
        if include_storefront and potential_brand:
            logger.info(f"Extracting storefront data for brand: {potential_brand}")

            storefront_query = f"amazon {potential_brand} brand store storefront page"

            payload["messages"][1]["content"] = f"Search for: {storefront_query}"

            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )

            if response.status_code == 200:
                response_data = response.json()
                tool_calls = response_data["choices"][0]["message"].get("tool_calls", [])
                if tool_calls:
                    search_result = tool_calls[0]["function"].get("arguments", "{}")
                    storefront_results = json.loads(search_result).get("search_results", "")

                    # Process storefront results
                    storefront_data = await _extract_storefront_data(potential_brand, storefront_results)
                    details["storefront_data"] = storefront_data
                    logger.info(f"Successfully extracted storefront data for {potential_brand}")

        # Step 7: Process all search results with GPT-4 to extract structured information
        combined_search_results = "\n\n---\n\n".join([result for result in search_results if result])

        extraction_prompt = f"""
        You are an expert Amazon product data extractor. Your task is to extract detailed, accurate product information from web search results about an Amazon product.

        Amazon Product URL: {url}
        {f"Amazon ASIN: {asin}" if asin else ""}
        {f"Product Name: {product_name}" if product_name else ""}
        {f"Brand Name: {brand_name}" if brand_name else ""}

        WEB SEARCH RESULTS:
        {combined_search_results}

        Extract the following information with high precision and detail:

        1. Product Title (full and accurate, including brand and model number if available)
        2. Brand Name (just the brand name without additional text like "by" or "from")
        3. Key Features/Bullet Points (extract at least 5-8 if available, focusing on technical specifications, materials, dimensions, and key selling points)
        4. Product Description (provide a detailed, comprehensive description that covers what the product is, how it works, and its benefits)
        5. Customer Reviews (extract 5-10 detailed reviews that provide specific feedback about the product, including both positive and negative perspectives)
        6. Questions and Answers (extract 3-5 detailed Q&A pairs that provide insights about common customer concerns or usage information)
        7. Technical Details (extract technical specifications in a structured format, including dimensions, weight, materials, compatibility, etc.)
        8. Enhanced Content (extract any A+ content, comparison charts, or brand story sections)

        IMPORTANT EXTRACTION GUIDELINES:
        - For the title, extract the EXACT product title as it appears on Amazon, including all model numbers and specifications
        - For bullet points, maintain the original formatting and technical details, avoiding generic marketing language
        - For the description, provide a comprehensive overview that includes technical details, use cases, and benefits
        - For reviews, include the reviewer's rating if available (e.g., "5-star review:") and focus on detailed, specific feedback
        - For Q&A, include complete questions and their corresponding answers
        - For technical details, organize them into a structured format with clear labels
        - For enhanced content, capture any special product presentations, comparison charts, or brand storytelling
        - If information is not available or unclear, indicate "Not Found" rather than guessing
        - Ensure all extracted information is specific to THIS product, not similar or related products

        Format your response as a valid JSON object with these keys:
        title, brand, bullets (array), description, reviews_raw (array), qna_raw (array), technical_details (object), enhanced_content.
        """

        # Call OpenAI with JSON response format
        extraction_response = call_openai_api(
            prompt=extraction_prompt,
            model=GPT_MODEL,
            temperature=0.2,  # Lower temperature for more factual extraction
            response_format={"type": "json_object"}
        )

        # Parse the response
        extracted_data = json.loads(extraction_response)

        # Update details with extracted data
        for key in extracted_data:
            if key in details and extracted_data[key]:
                details[key] = extracted_data[key]

        # Add additional fields if they exist in extracted data
        if "technical_details" in extracted_data and extracted_data["technical_details"]:
            details["technical_details"] = extracted_data["technical_details"]

        if "enhanced_content" in extracted_data and extracted_data["enhanced_content"]:
            details["enhanced_content"] = extracted_data["enhanced_content"]

        # Clean and validate the data
        details = _clean_amazon_data(details)

        # Cache the results
        CACHE[cache_key] = details
        logger.info(f"Updated cache for {cache_key}")
        return details

    except Exception as e:
        logger.error(f"Error during web search-based Amazon scraping for {url}: {e}")

        # Try a simpler fallback approach if the main method fails
        try:
            logger.info(f"Attempting simplified web search fallback for {url}")

            # Create a simpler query from the URL
            simple_query = f"Amazon product {asin if asin else product_name} details reviews questions"

            simple_prompt = f"""
            I need comprehensive information about this Amazon product:
            URL: {url}
            {f"ASIN: {asin}" if asin else ""}
            {f"Product Name: {product_name}" if product_name else ""}

            Please search for this product and provide the following details in a structured format:
            1. Complete Product Title
            2. Brand Name
            3. Key Features (at least 5 bullet points)
            4. Detailed Product Description (at least 150 words)
            5. Customer Reviews (at least 5 detailed reviews)
            6. Customer Questions and Answers (at least 3 Q&A pairs)
            7. Technical Details (specifications, dimensions, materials)
            8. Enhanced Content (A+ content or brand storytelling elements)

            For each piece of information, provide as much detail as possible. If you can't find certain information, indicate "Not Found" for that field.

            Format as JSON with these keys: title, brand, bullets (array), description, reviews_raw (array), qna_raw (array), technical_details (object), enhanced_content.
            """

            simple_response = call_openai_api(
                prompt=simple_prompt,
                model=GPT_MODEL,
                temperature=0.2,
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
        CACHE[cache_key] = details
        logger.info(f"Updated cache for {cache_key}")
        return details

async def _extract_storefront_data(brand_name: str, storefront_results: str) -> dict:
    """
    Extract Amazon storefront information for a brand

    Args:
    brand_name: The brand name to extract storefront data for
    storefront_results: Search results related to the brand storefront

    Returns:
    Dictionary containing structured storefront information
    """
    logger.info(f"Processing storefront data for brand: {brand_name}")

    # Find potential storefront URL
    storefront_url = None
    for line in storefront_results.split('\n'):
        if "amazon.com/stores" in line or "amazon.com/shop" in line:
            url_match = re.search(r'https?://(?:www\.)?amazon\.com/(?:stores|shop)/[^\s"\']+', line)
            if url_match:
                storefront_url = url_match.group(0)
                break

    storefront_prompt = f"""
    You are an expert Amazon brand storefront analyst. Extract comprehensive information about this brand's Amazon storefront.

    Brand Name: {brand_name}
    Storefront URL: {storefront_url if storefront_url else "Not explicitly found in search results"}

    SEARCH RESULTS:
    {storefront_results}

    Extract the following information about the brand's Amazon storefront:

    1. Brand Overview: A comprehensive summary of what the brand is about, its positioning, and its market focus
    2. Brand Mission/Values: Any stated mission, values, or commitments the brand makes
    3. Featured Products: Key product categories or flagship products highlighted in the storefront
    4. Brand Story: Any narrative about the brand's history, founding, or journey
    5. Visual Elements: Description of the brand's visual identity, imagery themes, and aesthetic
    6. Special Promotions: Any current promotions, deals, or special offerings
    7. Customer Testimonials: Any highlighted customer feedback or testimonials specific to the brand
    8. Brand Differentiators: What makes this brand stand out from competitors based on their storefront

    If certain information isn't available in the search results, indicate "Not Found" for that section.
    Format your response as a valid JSON object with these keys:
    overview, mission_values, featured_products (array), brand_story, visual_elements, promotions, testimonials (array), differentiators (array).
    """

    try:
        storefront_response = call_openai_api(
            prompt=storefront_prompt,
            model=GPT_MODEL,
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        storefront_data = json.loads(storefront_response)

        # Add the storefront URL if found
        if storefront_url:
            storefront_data["storefront_url"] = storefront_url

        return storefront_data

    except Exception as e:
        logger.error(f"Error extracting storefront data: {e}")
        return {
            "overview": "Not Found",
            "mission_values": "Not Found",
            "featured_products": [],
            "brand_story": "Not Found",
            "visual_elements": "Not Found",
            "promotions": "Not Found",
            "testimonials": [],
            "differentiators": [],
            "storefront_url": storefront_url if storefront_url else "Not Found"
        }

def _clean_amazon_data(details: dict) -> dict:
    """
    Clean and validate Amazon product data

    Args:
    details: Raw extracted product details

    Returns:
    Cleaned and validated product details
    """
    # Clean bullet points - remove duplicates and empty entries
    if "bullets" in details and isinstance(details["bullets"], list):
        cleaned_bullets = []
        seen_bullets = set()
        for bullet in details["bullets"]:
            # Normalize bullet text for deduplication
            normalized = re.sub(r'\s+', ' ', str(bullet).lower().strip())
            if normalized and normalized not in seen_bullets and len(normalized) > 5:
                cleaned_bullets.append(bullet)
                seen_bullets.add(normalized)
        details["bullets"] = cleaned_bullets

    # Clean reviews - remove duplicates and ensure they're substantial
    if "reviews_raw" in details and isinstance(details["reviews_raw"], list):
        cleaned_reviews = []
        seen_reviews = set()
        for review in details["reviews_raw"]:
            # Normalize review text for deduplication
            normalized = re.sub(r'\s+', ' ', str(review).lower().strip())
            if normalized and normalized not in seen_reviews and len(normalized) > 20:
                cleaned_reviews.append(review)
                seen_reviews.add(normalized)
        details["reviews_raw"] = cleaned_reviews

    # Clean Q&A - ensure they're in Q&A format and substantial
    if "qna_raw" in details and isinstance(details["qna_raw"], list):
        cleaned_qna = []
        for qna in details["qna_raw"]:
            if ("Q:" in str(qna) or "Question:" in str(qna)) and ("A:" in str(qna) or "Answer:" in str(qna)) and len(str(qna)) > 20:
                cleaned_qna.append(qna)
            elif "?" in str(qna) and len(str(qna)) > 20:
                # Try to format as Q&A if it contains a question mark
                parts = str(qna).split("?", 1)
                if len(parts) == 2:
                    formatted_qna = f"Q: {parts[0]}? A: {parts[1].strip()}"
                    cleaned_qna.append(formatted_qna)
        details["qna_raw"] = cleaned_qna

    return details

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
                "tools": [{
                    "type": "Web Search",
                    "function": {
                        "name": "Web Search",
                        "parameters": {}
                    }
                }],
                "tool_choice": {"type": "function", "function": {"name": "Web Search"}}
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
                    search_result = tool_calls[0]["function"].get("arguments", "{}")
                    search_results.append(json.loads(search_result).get("search_results", ""))

        # Second search: General product information if we don't have enough yet
        if not amazon_url or details["title"] == "Not Found" or not details["bullets"]:
            general_query = f"{query} product details specifications"
            payload = {
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": "You are a helpful web search assistant."},
                    {"role": "user", "content": f"Search for: {general_query}"}
                ],
                "tools": [{
                    "type": "Web Search",
                    "function": {
                        "name": "Web Search",
                        "parameters": {}
                    }
                }],
                "tool_choice": {"type": "function", "function": {"name": "Web Search"}}
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
                    search_result = tool_calls[0]["function"].get("arguments", "{}")
                    search_results.append(json.loads(search_result).get("search_results", ""))

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
                    search_result = tool_calls[0]["function"].get("arguments", "{}")
                    search_results.append(json.loads(search_result).get("search_results", ""))

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
                    search_result = tool_calls[0]["function"].get("arguments", "{}")
                    search_results.append(json.loads(search_result).get("search_results", ""))

        # Combine all search results
        combined_search_results = "\n\n---\n\n".join([result for result in search_results if result])

        # Process the combined search results to extract structured information
        extraction_prompt = f"""
        You are an expert product information extractor. Your task is to extract detailed, accurate product information from web search results.

        SEARCH QUERY: {query}
        {f"AMAZON URL: {amazon_url}" if amazon_url else ""}

        WEB SEARCH RESULTS:
        {combined_search_results}

        Extract the following information with high precision and detail:

        1. Product Title (full and accurate, including brand and model number if available)
        2. Brand Name (just the brand name without additional text like "by" or "from")
        3. Key Features/Bullet Points (extract at least 5-8 if available, focusing on technical specifications, materials, dimensions, and key selling points)
        4. Product Description (provide a detailed, comprehensive description that covers what the product is, how it works, and its benefits)
        5. Customer Reviews (extract 5-10 detailed reviews that provide specific feedback about the product, including both positive and negative perspectives)
        6. Questions and Answers (extract 3-5 detailed Q&A pairs that provide insights about common customer concerns or usage information)

        IMPORTANT EXTRACTION GUIDELINES:
        - For the title, extract the EXACT product title as it appears on the website
        - For bullet points, maintain the original formatting and technical details, avoiding generic marketing language
        - For the description, provide a comprehensive overview that includes technical details, use cases, and benefits
        - For reviews, include the reviewer's rating if available (e.g., "5-star review:") and focus on detailed, specific feedback
        - For Q&A, include complete questions and their corresponding answers
        - If information is not available or unclear, indicate "Not Found" rather than guessing

        Format your response as a valid JSON object with these keys: title, brand, bullets (array), description, reviews_raw (array), qna_raw (array).
        """

        # Call OpenAI with JSON response format
        extraction_response = call_openai_api(
            prompt=extraction_prompt,
            model=GPT_MODEL,
            temperature=0.2,  # Lower temperature for more factual extraction
            response_format={"type": "json_object"}
        )

        # Parse the response
        extracted_data = json.loads(extraction_response)

        # Validate and clean the extracted data
        if "title" in extracted_data and extracted_data["title"] and extracted_data["title"] != "Not Found":
            details["title"] = extracted_data["title"]

        if "brand" in extracted_data and extracted_data["brand"] and extracted_data["brand"] != "Not Found":
            details["brand"] = extracted_data["brand"]

        if "bullets" in extracted_data and isinstance(extracted_data["bullets"], list) and extracted_data["bullets"]:
            # Clean bullet points - remove duplicates and empty entries
            cleaned_bullets = []
            seen_bullets = set()
            for bullet in extracted_data["bullets"]:
                # Normalize bullet text for deduplication
                normalized = re.sub(r'\s+', ' ', bullet.lower().strip())
                if normalized and normalized not in seen_bullets and len(normalized) > 5:
                    cleaned_bullets.append(bullet)
                    seen_bullets.add(normalized)
            details["bullets"] = cleaned_bullets

        if "description" in extracted_data and extracted_data["description"] and extracted_data["description"] != "Not Found":
            details["description"] = extracted_data["description"]

        if "reviews_raw" in extracted_data and isinstance(extracted_data["reviews_raw"], list) and extracted_data["reviews_raw"]:
            # Clean reviews - remove duplicates and ensure they're substantial
            cleaned_reviews = []
            seen_reviews = set()
            for review in extracted_data["reviews_raw"]:
                # Normalize review text for deduplication
                normalized = re.sub(r'\s+', ' ', review.lower().strip())
                if normalized and normalized not in seen_reviews and len(normalized) > 20:
                    cleaned_reviews.append(review)
                    seen_reviews.add(normalized)
            details["reviews_raw"] = cleaned_reviews

        if "qna_raw" in extracted_data and isinstance(extracted_data["qna_raw"], list) and extracted_data["qna_raw"]:
            # Clean Q&A - ensure they're in Q&A format and substantial
            cleaned_qna = []
            for qna in extracted_data["qna_raw"]:
                if ("Q:" in qna or "Question:" in qna) and ("A:" in qna or "Answer:" in qna) and len(qna) > 20:
                    cleaned_qna.append(qna)
                elif "?" in qna and len(qna) > 20:
                    # Try to format as Q&A if it contains a question mark
                    parts = qna.split("?", 1)
                    if len(parts) == 2:
                        formatted_qna = f"Q: {parts[0]}? A: {parts[1].strip()}"
                        cleaned_qna.append(formatted_qna)
            details["qna_raw"] = cleaned_qna

        logger.info("Successfully extracted structured product information from web search results")
        return details

    except Exception as e:
        logger.error(f"Error during enhanced web search fallback: {e}")
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
    
    # Extract enhanced content if available
    amazon_enhanced = amazon_details.get("enhanced_content", "N/A")
    
    # Extract technical details if available
    amazon_technical = json.dumps(amazon_details.get("technical_details", {}))
    
    # Extract storefront data if available
    storefront_data = amazon_details.get("storefront_data", None)
    storefront_section = ""
    if storefront_data:
        storefront_section = f"""
        ### Amazon Brand Storefront Data:
        **Brand Overview:** {storefront_data.get('overview', 'Not Found')}
        **Brand Mission/Values:** {storefront_data.get('mission_values', 'Not Found')}
        **Brand Story:** {storefront_data.get('brand_story', 'Not Found')}
        **Featured Products:** {', '.join(storefront_data.get('featured_products', ['Not Found']))}
        **Visual Elements:** {storefront_data.get('visual_elements', 'Not Found')}
        **Brand Differentiators:** {', '.join(storefront_data.get('differentiators', ['Not Found']))}
        """

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
1. Scraped Amazon Listing Data (Title, Brand, Bullets, Description, Reviews, Q&A, Technical Details, Enhanced Content)
2. Amazon Brand Storefront Data (if available)
3. User Provided Input (Brand details, product category, challenges)
4. Scraped Website Content (supplementary context only)
5. Web search data (if scraping methods failed)

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
**Technical Details:**
{amazon_technical[:1000]}...
**Enhanced Content:**
{amazon_enhanced[:1000]}...
**Customer Reviews:**
{amazon_reviews[:1500]}...
**Customer Q&A:**
{amazon_qna[:1500]}...
**Customer Insights (VOC Analysis):**
{voc_insights}
{storefront_section}
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
def process_answer_text(text):
    """Process answer text to handle bullet points and formatting properly"""
    if not text:
        return ""

    # Replace markdown-style bullet points with HTML bullets
    lines = text.split('\n')
    processed_lines = []

    for line in lines:
        # Handle bullet points (*, -, •)
        line = line.strip()
        if line.startswith('* ') or line.startswith('- ') or line.startswith('• '):
            # Convert to HTML bullet
            processed_line = f"&#8226; {line[2:].strip()}"
            processed_lines.append(processed_line)
        elif line.startswith('1. ') or line.startswith('2. ') or line.startswith('3. '):
            # Handle numbered lists
            num, content = line.split('. ', 1)
            processed_line = f"{num}. {content.strip()}"
            processed_lines.append(processed_line)
        else:
            processed_lines.append(line)

    # Join lines with HTML line breaks
    processed_text = "<br/>".join(processed_lines)

    # Handle bold text (** or __)
    processed_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', processed_text)
    processed_text = re.sub(r'__(.*?)__', r'<b>\1</b>', processed_text)

    # Handle italic text (* or _)
    processed_text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', processed_text)
    processed_text = re.sub(r'_(.*?)_', r'<i>\1</i>', processed_text)

    return processed_text
    
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
            # Create a custom canvas class for page numbering
            class NumberedCanvas(Canvas):
                def __init__(self, *args, **kwargs):
                    Canvas.__init__(self, *args, **kwargs)
                    self._saved_page_states = []

                def showPage(self):
                    self._saved_page_states.append(dict(self.__dict__))
                    self._startPage()

                def save(self):
                    """Add page numbers to each page"""
                    num_pages = len(self._saved_page_states)
                    for state in self._saved_page_states:
                        self.__dict__.update(state)
                        self.setFont("Helvetica", 9)
                        self.setFillColor(colors.white)
                        self.drawRightString(
                            self._pagesize[0] - 0.5*inch,
                            0.5*inch,
                            f"Page {self._pageNumber} of {num_pages}"
                        )
                        Canvas.showPage(self)
                    Canvas.save(self)

            # Create document with better margins
            doc = SimpleDocTemplate(
                pdf_path,
                pagesize=LETTER,
                rightMargin=inch*0.75,
                leftMargin=inch*0.75,
                topMargin=inch*0.75,
                bottomMargin=inch*0.75
            )

            story = []
            styles = getSampleStyleSheet()

            # Create better styled paragraph formats with white text for black background
            title_style = ParagraphStyle(
                name='TitleStyle',
                parent=styles['Title'],
                fontSize=20,
                spaceAfter=16,
                fontName='Helvetica-Bold',
                alignment=1,  # Center alignment
                textColor=colors.white
            )

            h1_style = ParagraphStyle(
                name='H1Style',
                parent=styles['Heading1'],
                fontSize=16,
                spaceAfter=14,
                spaceBefore=20,
                fontName='Helvetica-Bold',
                textColor=colors.white
            )

            h2_style = ParagraphStyle(
                name='H2Style',
                parent=styles['Heading2'],
                fontSize=14,
                spaceAfter=10,
                spaceBefore=16,
                fontName='Helvetica-Bold',
                textColor=colors.white
            )

            question_style = ParagraphStyle(
                name='QuestionStyle',
                parent=styles['Normal'],
                fontSize=12,
                fontName='Helvetica-Bold',
                spaceAfter=4,
                spaceBefore=10,
                textColor=colors.white
            )

            answer_style = ParagraphStyle(
                name='AnswerStyle',
                parent=styles['Normal'],
                fontSize=11,
                fontName='Helvetica',
                leftIndent=20,
                spaceAfter=12,
                leading=14,  # Line spacing
                textColor=colors.white
            )

            # Special style for target customer section
            target_customer_style = ParagraphStyle(
                name='TargetCustomerStyle',
                parent=styles['Normal'],
                fontSize=11,
                fontName='Helvetica',
                spaceAfter=2,
                textColor=colors.white
            )

            # Add title page
            story.append(Paragraph(f"Creative Brief: {project_name}", title_style))
            story.append(Spacer(1, 0.3 * inch))

            # Add date
            from datetime import datetime
            current_date = datetime.now().strftime("%B %d, %Y")
            date_style = ParagraphStyle(
                name='DateStyle',
                parent=styles['Normal'],
                fontSize=11,
                alignment=1,  # Center alignment
                textColor=colors.white
            )
            story.append(Paragraph(f"Generated on {current_date}", date_style))

            # Process assets section
            inputs = json.loads(session.brand_input) if session.brand_input else {}

            # Section counter for numbering
            section_number = 1

            # Assets Overview section
            story.append(Paragraph(f"{section_number}. {MAIN_HEADINGS['ASSETS_OVERVIEW']}", h1_style))
            section_number += 1

            if inputs.get("has_assets", True):
                asset_fields = [
                    ("White Background Image Link", "white_background_image"),
                    ("Old Images Link", "old_images"),
                    ("Lifestyle Image Link", "lifestyle_image"),
                    ("User-Generated Content Link", "user_generated_content"),
                    ("Video Content Link", "video_content"),
                ]

                any_asset = False
                question_number = 1
                for label, key in asset_fields:
                    value = inputs.get(key, "")
                    if value:
                        any_asset = True
                        story.append(Paragraph(f"{section_number-1}.{question_number}. <b>{label}:</b>", question_style))
                        story.append(Paragraph(f"{value}", answer_style))
                        question_number += 1

                if not any_asset:
                    story.append(Paragraph("No asset links provided.", answer_style))
            else:
                story.append(Paragraph("No assets provided.", answer_style))

            story.append(Spacer(1, 0.2 * inch))

            # Process each section with better formatting
            sections = json.loads(session.form_data) if session.form_data else []

            for section in sections:
                heading = section.get("title", "Untitled Section")

                # Check if this is a main heading
                if heading in MAIN_HEADINGS.values():
                    story.append(Paragraph(f"{section_number}. {heading}", h1_style))

                    # Special handling for TARGET CUSTOMER DEEP DIVE section
                    if heading == "TARGET CUSTOMER DEEP DIVE":
                        # Extract demographic information from questions
                        demographics = {}
                        if "questions" in section and isinstance(section["questions"], list):
                            for qa in section["questions"]:
                                q = qa.get("question", "").lower()
                                a = qa.get("answer", "")

                                if "gender" in q:
                                    demographics["gender"] = a
                                elif "age" in q:
                                    demographics["age_range"] = a
                                elif "location" in q:
                                    demographics["location"] = a
                                elif "income" in q:
                                    demographics["income"] = a
                                elif "profession" in q:
                                    demographics["profession"] = a

                        # Add formatted demographic information
                        story.append(Paragraph(f"Gender = {demographics.get('gender', 'male and female')}", target_customer_style))
                        story.append(Paragraph(f"age range = {demographics.get('age_range', 'aged 25-45')}", target_customer_style))
                        story.append(Paragraph(f"location = {demographics.get('location', 'United States')}", target_customer_style))
                        story.append(Paragraph(f"income = {demographics.get('income', 'high income')}", target_customer_style))
                        story.append(Paragraph(f"profession = {demographics.get('profession', 'engaged in regular tennis activities')}", target_customer_style))

                        # Add other questions if any
                        question_number = 1
                        if "questions" in section and isinstance(section["questions"], list):
                            for qa in section["questions"]:
                                q = qa.get("question", "").lower()
                                a = qa.get("answer", "N/A")

                                # Skip demographic questions already handled
                                if any(term in q for term in ["gender", "age", "location", "income", "profession"]):
                                    continue

                                if q:
                                    story.append(Paragraph(f"{section_number}.{question_number}. {qa.get('question')}", question_style))
                                    processed_answer = process_answer_text(a)
                                    story.append(Paragraph(processed_answer, answer_style))
                                    question_number += 1
                    else:
                        # Regular section processing with numbered questions
                        question_number = 1
                        if "questions" in section and isinstance(section["questions"], list):
                            for qa in section["questions"]:
                                q = qa.get("question", "")
                                a = qa.get("answer", "N/A")

                                if q:
                                    story.append(Paragraph(f"{section_number}.{question_number}. {q}", question_style))
                                    processed_answer = process_answer_text(a)
                                    story.append(Paragraph(processed_answer, answer_style))
                                    question_number += 1

                    section_number += 1
                else:
                    # For non-main headings
                    story.append(Paragraph(heading, h2_style))

                    if "questions" in section and isinstance(section["questions"], list):
                        for qa in section["questions"]:
                            q = qa.get("question", "")
                            a = qa.get("answer", "N/A")

                            if q:
                                story.append(Paragraph(q, question_style))
                                processed_answer = process_answer_text(a)
                                story.append(Paragraph(processed_answer, answer_style))

                story.append(Spacer(1, 0.2 * inch))

            # Function to draw the black background
            def on_page(canvas, doc):
                canvas.saveState()
                canvas.setFillColor(colors.black)
                canvas.rect(0, 0, doc.pagesize[0], doc.pagesize[1], fill=1)
                canvas.restoreState()

            # Build the PDF with black background and page numbers
            doc.build(story, canvasmaker=NumberedCanvas, onFirstPage=on_page, onLaterPages=on_page)
            logger.info(f"PDF generated at {pdf_path} for session {session_id}.")

        except Exception as e:
            logger.error(f"Error generating PDF for session {session_id}: {e}")
            logger.exception(e)  # Log the full traceback

    return pdf_path

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
    include_storefront: bool = Form(False),  # New parameter for storefront extraction
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
        "no_assets_text": no_assets_text, "include_storefront": include_storefront
    }
    amazon_details = await scrape_amazon_listing_details(amazon_listing, include_storefront) if amazon_listing else {}
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