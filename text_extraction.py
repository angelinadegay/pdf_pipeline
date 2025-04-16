import asyncio
from collections import defaultdict
import re
import tiktoken
import time
from openai import OpenAI
import nltk
import argparse
import json
from pathlib import Path
import os
from dotenv import load_dotenv
import sys
import logging
from tqdm import tqdm
import yaml
from datetime import datetime
import PyPDF2
import io

from nltk.tokenize import sent_tokenize

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"text_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load configuration
def load_config(config_file="config.yaml"):
    """Load configuration from YAML file."""
    default_config = {
        "api": {
            "model": "gpt-3.5-turbo",
            "max_tokens": 3000,
            "retries": 5,
            "retry_delay": 1
        },
        "processing": {
            "chunk_size": 3000,
            "max_parallel_tasks": 5
        },
        "output": {
            "default_format": "txt",
            "include_timestamps": True
        }
    }
    
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            # Merge with default config
            return {**default_config, **config}
    return default_config

# Load configuration
config = load_config()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Extract themed quotations from construction documentation.')
    parser.add_argument('--input-file', type=str, required=True, help='Path to the input text or PDF file')
    parser.add_argument('--output-file', type=str, default='extracted_quotes.txt', help='Path to the output file')
    parser.add_argument('--section-title', type=str, required=True, help='Title of the section to extract (e.g., "SECTION 230913 - INSTRUMENTATION AND CONTROL FOR HVAC EQUIPMENT")')
    parser.add_argument('--themes-file', type=str, help='JSON file containing themes and their keywords')
    parser.add_argument('--config-file', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--format', type=str, choices=['txt', 'json', 'csv'], default=config['output']['default_format'], help='Output format')
    return parser.parse_args()

def load_themes(themes_file):
    """
    Load themes from a JSON file or use default themes if the file doesn't exist.
    
    Args:
        themes_file: Path to the JSON file containing themes and their keywords
        
    Returns:
        Dictionary of themes and their keywords
    """
    # Default themes if no file is provided or file doesn't exist
    default_themes = {
        "Graphics, Software": ["graphics", "GUI"],
        "Mention of ATC Systems, Metasys, JohnsonFX": ["ATC Systems", "Metasys", "JohnsonFX"],
        "Project Management": ["project management", "schedule", "deadline"],
        "Protocols": ["protocol", "standards", "communication"],
        "Network Automation Engine": ["network automation engine"],
        "Sensors": ["sensor", "detector", "measurement"],
        "Smoke, Fire Alarms, Detectors": ["smoke", "fire alarm", "detector"],
        "System Devices": ["system device", "component"],
        "Enclosures/Covers": ["enclosure", "cover"],
        "BMS": ["BMS", "building management system"],
        "Wiring": ["wiring", "cable", "electrical"],
        "Controls": ["control", "controller"],
        "Mechanical Contractor and Installation": ["mechanical contractor", "installation"],
        "Thermostat/Temperature": ["thermostat", "temperature"],
        "Sequence of Operations": ["sequence of operations"],
    }
    
    # If no themes file is specified, use default themes
    if not themes_file:
        logger.info("No themes file specified. Using default themes.")
        return default_themes
    
    # Try to load themes from the specified file
    try:
        if Path(themes_file).exists():
            with open(themes_file, 'r') as f:
                themes = json.load(f)
                logger.info(f"Loaded {len(themes)} themes from {themes_file}")
                return themes
        else:
            logger.warning(f"Themes file '{themes_file}' not found. Using default themes.")
            return default_themes
    except Exception as e:
        logger.error(f"Error loading themes from {themes_file}: {str(e)}")
        logger.info("Using default themes instead.")
        return default_themes

def convert_pdf_to_text(pdf_path):
    """
    Convert a PDF file to text.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text from the PDF
    """
    logger.info(f"Converting PDF to text: {pdf_path}")
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            
            # Use tqdm for progress tracking
            for page_num in tqdm(range(total_pages), desc="Extracting PDF pages"):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n\n"
                
        logger.info(f"Successfully extracted text from {total_pages} pages")
        return text
    except Exception as e:
        logger.error(f"Error converting PDF to text: {str(e)}")
        raise

def read_input_file(file_path):
    """
    Read text from a file, handling both text and PDF files.
    
    Args:
        file_path: Path to the input file
        
    Returns:
        Text content from the file
    """
    file_extension = Path(file_path).suffix.lower()
    
    # Only process as PDF if the file has a .pdf extension
    if file_extension == '.pdf':
        logger.info(f"Processing file as PDF: {file_path}")
        return convert_pdf_to_text(file_path)
    else:
        # Process as text file for all other extensions
        logger.info(f"Processing file as text: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error reading text file: {str(e)}")
            raise
    # Get API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.error("OPENAI_API_KEY environment variable not set. Please create a .env file with your API key.")
    exit(1)
    
    # Initialize OpenAI client
client = OpenAI(api_key=api_key)
gpt_model = "gpt-3.5-turbo"

def extract_section(text, section_title):
    pattern = rf"({section_title}.*?END OF SECTION \d+)"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1)
    else:
        return None

def preprocess_text(text):
    """
    Preprocess the text to normalize lists and bullet points.
    This helps maintain context when splitting into chunks.
    """
    # Replace numbered lists with a consistent format
    text = re.sub(r'(\d+\.\s+)', r'<num>\1', text)
    
    # Replace lettered lists with a consistent format
    text = re.sub(r'([a-z]\.\s+)', r'<letter>\1', text)
    
    # Replace bullet points with a consistent format
    text = re.sub(r'(\*\s+)', r'<bullet>\1', text)
    text = re.sub(r'(-\s+)', r'<bullet>\1', text)
    
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text

def postprocess_text(text):
    """
    Restore the original format of lists and bullet points.
    """
    # Restore numbered lists
    text = re.sub(r'<num>(\d+\.\s+)', r'\1', text)
    
    # Restore lettered lists
    text = re.sub(r'<letter>([a-z]\.\s+)', r'\1', text)
    
    # Restore bullet points
    text = re.sub(r'<bullet>(\*\s+)', r'\1', text)
    text = re.sub(r'<bullet>(-\s+)', r'\1', text)
    
    return text

def split_text_into_chunks(text, max_tokens=3000):
    """
    Split text into chunks while preserving list context.
    """
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    encoding = tiktoken.get_encoding("cl100k_base")
    
    # Split by paragraphs first to avoid breaking lists
    paragraphs = re.split(r'\n\s*\n', processed_text)
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for paragraph in paragraphs:
        paragraph_tokens = encoding.encode(paragraph)
        
        # If a single paragraph exceeds max_tokens, split it by sentences
        if len(paragraph_tokens) > max_tokens:
            sentences = sent_tokenize(postprocess_text(paragraph))
            for sentence in sentences:
                sentence_tokens = encoding.encode(sentence)
                
                if current_tokens + len(sentence_tokens) > max_tokens:
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))
                        current_chunk = []
                        current_tokens = 0
                
                current_chunk.append(sentence)
                current_tokens += len(sentence_tokens)
        else:
            # If adding this paragraph would exceed max_tokens, start a new chunk
            if current_tokens + len(paragraph_tokens) > max_tokens:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
            
            current_chunk.append(postprocess_text(paragraph))
            current_tokens += len(paragraph_tokens)
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def combine_results(all_chunks):
    combined = defaultdict(list)
    for chunk in all_chunks:
        if not isinstance(chunk, dict):  # Ensure chunk is valid
            continue
        for theme, quotes in chunk.items():
            if quotes:
                combined[theme].extend(quotes)
    return combined

def parse_gpt_output(raw_output):
    parsed = defaultdict(list)
    lines = raw_output.splitlines()
    current_theme = None
    for line in lines:
        line = line.strip()
        if line.startswith("**") and line.endswith(":**"):  # Theme header
            current_theme = line.strip("**:").strip()
        elif line.startswith("-") and current_theme:
            if "No specific information found" not in line:# Quotation under the theme
                parsed[current_theme].append(line.lstrip("- ").strip())
    return parsed

async def query_gpt_with_retry(section_title, section_content, themes, model=gpt_model, retries=5):
    theme_list = "\n".join([f"- {theme}" for theme in themes])
    prompt = f"""
You are an assistant that identifies and extracts direct quotations relevant to specific themes from the text.
For the section titled "{section_title}", do the following:

1. Identify and list direct quotations grouped by the following themes:
   {theme_list}

2. For list items (numbered or lettered), include the entire list item as a single quotation if it's relevant to a theme.

3. If no information is found for a theme, do not include that theme in the response, but do not write anything like this: " No specific information found in the provided text for this theme.", just don't say anything.

Text:
{section_content}
"""

    print(f"Prompt: {prompt}")  # For debugging
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an assistant that extracts direct quotations by theme."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            raw_output = response.choices[0].message.content
            dict_output = parse_gpt_output(raw_output)
            return dict_output
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            time.sleep(1)
    return {}

async def process_chunk_with_retry(chunk_index, chunk, themes, delay=config['api']['retry_delay']):
    """Process a chunk with retry logic and progress tracking."""
    for attempt in range(config['api']['retries']):
        try:
            logger.info(f"Processing chunk {chunk_index + 1}, attempt {attempt + 1}/{config['api']['retries']}")
            quotes = await query_gpt_with_retry(f"SECTION 230913 - Chunk {chunk_index + 1}", chunk, themes)
            return quotes
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_index + 1}, attempt {attempt + 1}: {str(e)}")
            if attempt < config['api']['retries'] - 1:
                logger.info(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
            else:
                logger.error(f"Failed to process chunk {chunk_index + 1} after {config['api']['retries']} attempts")
                return {}

async def main():
    args = parse_arguments()
    
    # Load configuration from file if specified
    if args.config_file:
        global config
        config = load_config(args.config_file)
    

    
    # Load themes
    themes = load_themes(args.themes_file)
    logger.info(f"Using {len(themes)} themes for extraction")
    
    # Read input file
    try:
        logger.info(f"Reading input file: {args.input_file}")
        text = read_input_file(args.input_file)
    except FileNotFoundError:
        logger.error(f"Input file '{args.input_file}' not found.")
        exit(1)
    except Exception as e:
        logger.error(f"Error reading input file: {str(e)}")
        exit(1)

    section = extract_section(text, args.section_title)
    if section:
        logger.info(f"Section '{args.section_title}' found. Splitting into chunks...")
        chunks = split_text_into_chunks(section, max_tokens=config['processing']['chunk_size'])
        logger.info(f"Split into {len(chunks)} chunks")

        # Initialize combined_results
        combined_results = defaultdict(list)
        
        # Process chunks with progress bar
        tasks = []
        for i, chunk in enumerate(chunks):
            task = process_chunk_with_retry(i, chunk, themes)
            tasks.append(task)
            
            # Limit parallel tasks
            if len(tasks) >= config['processing']['max_parallel_tasks']:
                results = await asyncio.gather(*tasks)
                tasks = []
                for result in results:
                    if result:
                        for theme, quotes in result.items():
                            combined_results[theme].extend(quotes)

        # Process remaining tasks
        if tasks:
            results = await asyncio.gather(*tasks)
            for result in results:
                if result:
                    for theme, quotes in result.items():
                        combined_results[theme].extend(quotes)

        # Write results to output file
        logger.info(f"Writing results to {args.output_file}")
        try:
            with open(args.output_file, "w", encoding="utf-8") as output_file:
                if args.format == 'json':
                    json.dump(combined_results, output_file, indent=2)
                elif args.format == 'csv':
                    import csv
                    writer = csv.writer(output_file)
                    writer.writerow(['Theme', 'Quote'])
                    for theme, quotes in combined_results.items():
                        for quote in quotes:
                            writer.writerow([theme, quote])
                else:  # txt format
                    for theme, quotes in combined_results.items():
                        if quotes:
                            output_file.write(f"**{theme}:**\n")
                            for quote in quotes:
                                output_file.write(f"- {quote}\n")
                            output_file.write("\n")
            
            logger.info(f"Results successfully written to {args.output_file}")
            
            # Print summary
            total_quotes = sum(len(quotes) for quotes in combined_results.values())
            logger.info(f"Extracted {total_quotes} quotes across {len(combined_results)} themes")
            
        except Exception as e:
            logger.error(f"Error writing output file: {str(e)}")
            exit(1)
    else:
        logger.error(f"Section '{args.section_title}' not found in the text.")

def run_default():
    """Run the program with default arguments."""
    # Set default arguments
    sys.argv = [
        sys.argv[0],
        "--input-file", "sample1.txt",
        "--section-title", "SECTION 230913 - INSTRUMENTATION AND CONTROL FOR HVAC EQUIPMENT",
        "--themes-file", "themes.json",
        "--output-file", "extracted_quotes.txt"
    ]
    
    # Run the main function
    asyncio.run(main())

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run with command line arguments
        asyncio.run(main())
    else:
        # Run with default arguments
        run_default()