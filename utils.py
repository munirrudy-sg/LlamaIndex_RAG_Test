import logging
import requests
from bs4 import BeautifulSoup
from llama_index.core import Document
import re
from typing import List
import time
from urllib.parse import urlparse

def fetch_html(url):
    """
    Fetch the HTML content of the given URL.
    """
    try:
        logging.info(f"Fetching HTML content from {url}")
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        return response #.text
    except requests.RequestException as e:
        logging.error(f"Error fetching the URL {url}: {e}")
        return None

def get_context(url):
    """
    Extracts the text content from a given HTML section.

    Args:
        html: A BeautifulSoup element representing the HTML section.

    Returns:
        The extracted text content as a string.
    """
    html = fetch_html(url)
    soup = BeautifulSoup(html.text, features="html.parser")
    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()    # rip it out

    # get text
    context = soup.get_text(separator=" ")

    return context

def preprocess(text):
    # Remove Unicode private use area characters
    cleaned_text = text.replace('|', ' ').replace(':', ' ').replace('?',' ').replace('-',' ')
    cleaned_text = re.sub(r'[\ue000-\uf8ff]', ' ', cleaned_text)
    # Remove extra whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    # Normalize whitespace but keep newlines intact
    # cleaned_text = re.sub(r' +', ' ', cleaned_text)  # Replace multiple spaces with a single space
    # cleaned_text = re.sub(r'\n\s+', '\n', cleaned_text)  # Normalize spaces after newlines
    cleaned_text = cleaned_text.strip()  # Remove leading and trailing spaces
    return cleaned_text.lower()

def clean_context(html_content):
    """
    Extracts the text content from a given HTML section.

    Args:
        html: A BeautifulSoup element representing the HTML section.

    Returns:
        The extracted text content as a string.
    """
    soup = BeautifulSoup(html_content, features="html.parser")
    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()    # rip it out

    # get text
    text = soup.get_text(separator=" ")

    # Handle plain text processing
    # formatted_text = re.sub(r"\s+", " ", text).strip()
    # cleaned_lines = [line.strip() for line in formatted_text.splitlines() if line.strip()]
    # cleaned_text = "\n".join(cleaned_lines)

    # Split the text into lines
    lines = text.split("\n")

    # Remove duplicate newlines
    cleaned_lines = []
    for line in lines:
        if line and line not in cleaned_lines:

            # # Remove extra spaces (including \xa0)
            line = re.sub(r"\s+", " ", line).strip()
            cleaned_lines.append(line)

    # Join the cleaned lines with a single newline
    cleaned_text = "\n".join(cleaned_lines)

    return cleaned_text

def extract_promotional_title(url):
    path = urlparse(url).path
    parts = path.split('/')
    parts = [part for part in parts if part and part not in ('id', 'promosi')]
    title = ' '.join(parts).replace('-', ' ').replace('diskon', '')
    return title if title else "semua promo"

def process_webpages(all_promo_webpages: List[str], detail_promo_webpages: List[str]) -> List[Document]:
  """
  Processes a list of webpages by extracting text content and storing them as documents.

  Args:
      webpages: A list of URLs or HTML strings representing webpages.

  Returns:
      A list of Document objects containing the extracted text content.
  """

  documents = []

  # All promo
  for webpage in all_promo_webpages:
    # Assuming you have a way to retrieve HTML content from URLs
    # Replace this with your logic to fetch HTML content
    if isinstance(webpage, str):  # If it's a URL
      html_content = get_context(webpage)  # Replace with your fetch function
    else:  # If it's already HTML content
      html_content = webpage

    if html_content:
      context = get_context(webpage)
      time.sleep(1)
      cleaned_context = clean_context(context)
      cleaned_text = preprocess(cleaned_context)
      final_text = extract_all_promo_text(cleaned_text)
      title = extract_promotional_title(webpage)
      print(title)
      documents.append(Document(text=final_text, 
                                metadata={'title': f'{title}'},
                                metadata_seperator="::",
                                metadata_template="{key}=>{value}",
                                text_template="Metadata: {metadata_str}\n-----\nContent: {content}"))
      
  # Detail promo
  for webpage in detail_promo_webpages:
    # Assuming you have a way to retrieve HTML content from URLs
    # Replace this with your logic to fetch HTML content
    if isinstance(webpage, str):  # If it's a URL
      html_content = get_context(webpage)  # Replace with your fetch function
    else:  # If it's already HTML content
      html_content = webpage

    if html_content:
      context = get_context(webpage)
      time.sleep(1)
      cleaned_context = clean_context(context)
      cleaned_text = preprocess(cleaned_context)
      final_text = extract_detail_promo_text(cleaned_text)
      title = extract_promotional_title(webpage)
      print(title)
      documents.append(Document(text=final_text, 
                                metadata={'title': f'{title}'},
                                metadata_seperator="::",
                                metadata_template="{key}=>{value}",
                                text_template="Metadata: {metadata_str}\n-----\nContent: {content}"))

  return documents

def extract_detail_promo_text(text):
    """
    Extracts the text content between "Semua Promo" and "GRATIS Pakai SimobiPlus" from an HTML page.

    Args:
        html_content (str): The HTML content as a string.

    Returns:
        str: The extracted text content.
    """
    # Find the starting index of the substring
    start_index = text.index("promosi") + len("promosi") + 1 # +1 for extra space

    # Find the ending index of the substring
    end_index = text.index("bagikan")

    # Extract the substring using string slicing
    extracted_text = text[start_index:end_index -1] #-1 to delete extra space

    return extracted_text

def extract_all_promo_text(text):
    """
    Extracts the text content between "Semua Promo" and "GRATIS Pakai SimobiPlus" from an HTML page.

    Args:
        html_content (str): The HTML content as a string.

    Returns:
        str: The extracted text content.
    """
    # Find the starting index of the substring
    start_index = text.index("semua promo")

    # Find the ending index of the substring
    end_index = text.index("kantor pusat")

    # Extract the substring using string slicing
    extracted_text = text[start_index:end_index]

    return extracted_text


