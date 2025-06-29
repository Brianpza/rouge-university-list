import os
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
from collections import defaultdict
import csv
import json

load_dotenv()  # This loads variables from .env into os.environ

openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Main folder name
main_folder = "universitylist"
# This will now store structured university data instead of just CSV paths
all_extracted_university_data = []

# List of country folder names
# Modified to only include Thailand for targeted processing
country_folders = [
    "Thailand"
]

# Simple region mapping for ASEAN countries
ASEAN_REGIONS = {
    "Brunei": "Southeast Asia",
    "Myanmar": "Southeast Asia",
    "Cambodia": "Southeast Asia",
    "Timor-Leste": "Southeast Asia",
    "Indonesia": "Southeast Asia",
    "Laos": "Southeast Asia",
    "Malaysia": "Southeast Asia",
    "Philippines": "Southeast Asia",
    "Singapore": "Southeast Asia",
    "Thailand": "Southeast Asia",
    "Vietnam": "Southeast Asia"
}

# Create main folder if it doesn't exist
try:
    os.makedirs(main_folder, exist_ok=True)
    print(f"Main folder '{main_folder}' created or already exists")
except Exception as e:
    print(f"Error creating main folder: {e}")
    exit()

# Create country folders inside the main folder
for country in country_folders:
    folder_path = os.path.join(main_folder, country)
    try:
        os.makedirs(folder_path, exist_ok=True)
        print(f"Created folder: {folder_path}")
    except Exception as e:
        print(f"Error creating folder {folder_path}: {e}")

print("\nFolder creation process completed.")
print(f"All country folders have been created inside '{main_folder}'")

def extract_university_tables_from_url(url, country_name):
    """
    Fetches and parses an HTML page from a URL to extract all tables with 
    the 'wikitable' class. It specifically extracts the hyperlink from the 'Website'
    column and returns the data as a list of dictionaries.

    Args:
        url (str): The URL of the webpage to be processed.
        country_name (str): The name of the country for the universities being extracted.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary represents a 
                    university and its extracted data, including 'country'.
    """
    extracted_data = []
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')
        tables = soup.find_all('table', class_='wikitable')

        if not tables:
            print("No tables with the class 'wikitable' were found on the webpage.")
            return extracted_data

        print(f"Found {len(tables)} tables to extract from {url}.")

        for i, table in enumerate(tables):
            table_headers = [header.get_text(strip=True) for header in table.find_all('th')]
            
            website_col_idx = -1
            try:
                website_col_idx = [h.lower() for h in table_headers].index('website')
            except ValueError:
                print(f"Warning: 'Website' column not found in table {i + 1} for {country_name}.")

            tbody = table.find('tbody')
            if not tbody:
                print(f"Warning: No tbody found in table {i + 1}. Skipping.")
                continue

            for row in tbody.find_all('tr'):
                cells = row.find_all('td')
                if not cells or len(cells) != len(table_headers):
                    continue

                row_data_dict = {}
                for idx, cell in enumerate(cells):
                    header = table_headers[idx]
                    if idx == website_col_idx:
                        link_tag = cell.find('a', href=True)
                        row_data_dict[header] = link_tag['href'] if link_tag else cell.get_text(strip=True)
                    else:
                        row_data_dict[header] = cell.get_text(strip=True)
                
                # Add country and clean up common university name columns
                row_data_dict['Country'] = country_name
                
                # Standardize university name column for easier access later
                university_name_found = False
                possible_name_columns = ['name', 'names', 'name in english', 'names in english', 
                                         'institution', 'institutions', 'university', 'universities']
                
                for col_key in possible_name_columns:
                    for header_orig in table_headers:
                        if col_key in header_orig.lower():
                            row_data_dict['University'] = row_data_dict.pop(header_orig) # Rename to 'University'
                            university_name_found = True
                            break
                    if university_name_found:
                        break
                
                if university_name_found:
                    extracted_data.append(row_data_dict)
                else:
                    print(f"Warning: No clear university name column found for a row in {country_name}. Skipping row.")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL {url}: {e}")
    except Exception as e:
        print(f"An error occurred during extraction from {url}: {e}")
    return extracted_data

def check_with_openai(university_name):
    """
    Checks with OpenAI if a university has an agriculture department.
    """
    prompt = f"Does {university_name} have an agriculture department or related program? Answer with just 'Yes' or 'No'."
    try:
        response = openai.chat.completions.create(
            model="gpt-4.1-mini-2025-04-14", # Using gpt-4 for better accuracy
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides accurate information about university departments."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0.0 # Make it deterministic
        )
        answer = response.choices[0].message.content.strip().lower()
        return 'yes' in answer
    except Exception as e:
        print(f"Error querying OpenAI for agriculture department for {university_name}: {e}")
        return False
    
def check_with_openai_TTO(university_name):
    """
    Checks with OpenAI if a university has a TTO/KTO office.
    """
    prompt = f"Does {university_name} have a Technology Transfer Office (TTO) or Knowledge Transfer Office (KTO) or a similar intellectual property commercialization office? Answer with just 'Yes' or 'No'."
    try:
        response = openai.chat.completions.create(
            model="gpt-4.1-mini-2025-04-14", # Using gpt-4 for better accuracy
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides accurate information about university offices."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0.0 # Make it deterministic
        )
        answer = response.choices[0].message.content.strip().lower()
        return 'yes' in answer
    except Exception as e:
        print(f"Error querying OpenAI for TTO for {university_name}: {e}")
        return False

def google_search_for_url(query, site_filter=None):
    """
    Performs a Google search and returns the first relevant URL.
    
    Args:
        query (str): The search query.
        site_filter (str, optional): A domain to restrict the search to (e.g., "site:university.edu").
    
    Returns:
        str: The URL of the first search result, or None if not found.
    """
    full_query = f"{query} {site_filter}" if site_filter else query
    print(f"Searching Google for: '{full_query}'")
    try:
        # Using a direct search URL and parsing to find the link
        search_url = f"https://www.google.com/search?q={requests.utils.quote(full_query)}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(search_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all search result links
        for link in soup.find_all('a', href=True):
            href = link['href']
            # Basic filtering for actual search results, avoiding google internal links
            if href.startswith('/url?q=') and 'webcache' not in href and 'accounts.google.com' not in href:
                actual_url = href.split('/url?q=')[1].split('&')[0]
                return actual_url
        return None
    except requests.exceptions.RequestException as e:
        print(f"Google search error for '{full_query}': {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during Google search for '{full_query}': {e}")
        return None

def get_tto_page_url(university_name, university_website):
    """
    Attempts to find the TTO page URL for a university.
    """
    # Try searching directly on the university's site first
    if university_website and not university_website.startswith("http"):
        university_website = f"http://{university_website}" # Add scheme if missing

    if university_website:
        domain = university_website.split('//')[-1].split('/')[0]
        site_filter = f"site:{domain}"
    else:
        site_filter = None

    queries = [
        f"{university_name} TTO office {site_filter}",
        f"{university_name} technology transfer office {site_filter}",
        f"{university_name} intellectual property commercialization {site_filter}",
        f"{university_name} KTO office {site_filter}",
        f"{university_name} research commercialization {site_filter}"
    ]
    
    for query in queries:
        url = google_search_for_url(query)
        if url:
            # Basic validation to ensure the URL is somewhat relevant
            if 'tto' in url.lower() or 'technology-transfer' in url.lower() or 'kto' in url.lower():
                return url
    return "Not Found"

def get_incubation_record(university_name, university_website):
    """
    Attempts to find information about incubation records or startup support.
    """
    if university_website and not university_website.startswith("http"):
        university_website = f"http://{university_website}" # Add scheme if missing

    if university_website:
        domain = university_website.split('//')[-1].split('/')[0]
        site_filter = f"site:{domain}"
    else:
        site_filter = None

    queries = [
        f"{university_name} incubation program {site_filter}",
        f"{university_name} startup accelerator {site_filter}",
        f"{university_name} entrepreneurship center {site_filter}",
        f"{university_name} student startups {site_filter}"
    ]

    for query in queries:
        url = google_search_for_url(query)
        if url:
            # A more detailed check might be needed here, but for now, just finding a relevant page
            if 'incubation' in url.lower() or 'startup' in url.lower() or 'entrepreneurship' in url.lower():
                return f"Found relevant page: {url}"
    
    # If no specific page found, try a broader search for news/articles
    broad_query = f"{university_name} innovation ecosystem OR startup success OR incubation achievements"
    broad_url = google_search_for_url(broad_query)
    if broad_url:
        return f"Broader information found: {broad_url}"
    
    return "Not Found/No specific record"

def find_university_linkedin(university_name):
    """
    Search for a university's LinkedIn page.
    """
    search_query = f"{university_name} site:linkedin.com/school"
    linkedin_url = google_search_for_url(search_query)
    
    if linkedin_url and 'linkedin.com/school/' in linkedin_url:
        return linkedin_url
    
    # Fallback to general LinkedIn search if direct school page not found
    linkedin_search = f"https://www.linkedin.com/search/results/all/?keywords={requests.utils.quote(university_name)}&origin=GLOBAL_SEARCH_HEADER&entityType=school"
    return linkedin_search

# --- Script Execution ---
if __name__ == "__main__":
    print("Starting data extraction from Wikipedia...")
    for country in country_folders: # This loop will now only run for "Thailand"
        wikipedia_url = ""
        if country == "Philippines": # This condition will not be met
            wikipedia_url = f"https://en.wikipedia.org/wiki/List_of_colleges_and_universities_in_the_Philippines"
        else:
            wikipedia_url = f"https://en.wikipedia.org/wiki/List_of_universities_in_{country}"
        
        print(f"\nProcessing {country} from {wikipedia_url}")
        
        # Get data for the current country
        country_universities_data = extract_university_tables_from_url(wikipedia_url, country)
        
        # Add to our global list of all universities, limited to 100
        all_extracted_university_data.extend(country_universities_data[:100])
    
    print(f"\nFinished initial data extraction. Total universities found (limited to 100 per country): {len(all_extracted_university_data)}")

    final_university_data = []
    processed_count = 0

    print("\nProcessing each university for detailed information and OpenAI checks...")
    for uni_info in all_extracted_university_data:
        if processed_count >= 20: # Enforce overall limit of 100 universities
            print("Reached maximum of 100 universities for detailed processing. Stopping.")
            break

        university_name = uni_info.get('University')
        if not university_name:
            print(f"Skipping a record due to missing 'University' name: {uni_info}")
            continue

        print(f"\n--- Processing: {university_name} ---")
        
        website = uni_info.get('Website', 'N/A')
        country = uni_info.get('Country', 'N/A')
        region = ASEAN_REGIONS.get(country, 'Unknown')

        has_agriculture = check_with_openai(university_name)
        
        if has_agriculture:
            print(f"  -> Has Agriculture Department: Yes")
            has_tto = check_with_openai_TTO(university_name)
            tto_page_url = "N/A"
            if has_tto:
                tto_page_url = get_tto_page_url(university_name, website)
                print(f"  -> Has TTO: Yes, TTO Page URL: {tto_page_url}")
            else:
                print(f"  -> Has TTO: No")

            incubation_record = get_incubation_record(university_name, website)
            print(f"  -> Incubation Record: {incubation_record}")
            
            linkedin_search_url = find_university_linkedin(university_name)
            print(f"  -> Apollo/LinkedIn Search URL: {linkedin_search_url}")

            final_university_data.append({
                'University': university_name,
                'Country': country,
                'Region': region,
                'Website': website,
                'Has TTO?': 'Yes' if has_tto else 'No',
                'TTO Page URL': tto_page_url,
                'Incubation Record': incubation_record,
                'Apollo/LinkedIn Search URL': linkedin_search_url
            })
            processed_count += 1 # Increment only if processed for final CSV
        else:
            print(f"  -> Has Agriculture Department: No (Skipping detailed processing)")
    
    output_csv_filename = "asean_universities_data_thailand_limited.csv" # Changed output filename
    if final_university_data:
        # Define the exact columns order
        headers = [
            'University', 'Country', 'Region', 'Website', 
            'Has TTO?', 'TTO Page URL', 'Incubation Record', 'Apollo/LinkedIn Search URL'
        ]
        
        with open(output_csv_filename, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(final_university_data)
        print(f"\nSuccessfully created '{output_csv_filename}' with {len(final_university_data)} records.")
    else:
        print("\nNo universities with agriculture departments were found to generate the CSV.")

    print("\nScript execution finished.")
