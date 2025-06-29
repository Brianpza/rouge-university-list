import streamlit as st
import pandas as pd
import io
import sys
import contextlib

# Import all functions from your backend script (assuming it's named main.py)
# If your backend script has a different name, replace 'main' with that name.
from main import (
    country_folders, ASEAN_REGIONS,
    extract_university_tables_from_url, check_with_openai, check_with_openai_TTO,
    google_search_for_url, get_tto_page_url, get_incubation_record,
    find_university_linkedin, OpenAI # Import OpenAI client to set API key
)

# Set up the Streamlit page
st.set_page_config(
    page_title="ASEAN University Data Extractor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎓 ASEAN University Data Extractor")
st.markdown("""
This application extracts information about universities in ASEAN countries,
specifically focusing on **Thailand**. You can specify the maximum number of universities
to process, up to 100. It identifies universities with agriculture departments,
Technology Transfer Offices (TTOs), incubation records, and provides LinkedIn search URLs.
""")

# OpenAI API Key Input and configurable limit
with st.sidebar:
    st.header("Configuration")
    openai_api_key = st.text_input(
        "Enter your OpenAI API Key",
        type="password",
        help="You can get your API key from https://platform.openai.com/account/api-keys"
    )
    st.info("Your API key is not stored and is only used for the current session.")

    if openai_api_key:
        st.success("OpenAI API Key provided.")
        try:
            from main import openai as backend_openai_client
            backend_openai_client.api_key = openai_api_key
            st.session_state['openai_configured'] = True
        except Exception as e:
            st.error(f"Failed to configure OpenAI client: {e}")
            st.session_state['openai_configured'] = False
    else:
        st.warning("Please enter your OpenAI API Key to proceed.")
        st.session_state['openai_configured'] = False

    # New: User-defined limit for universities
    university_limit = st.number_input(
        "Maximum Universities to Process (Thailand)",
        min_value=1,
        max_value=100,
        value=10, # Default value for testing
        step=1,
        help="Set the maximum number of universities to process for detailed information. Max is 100."
    )
    # Store the limit in session state for consistency
    st.session_state['university_limit'] = university_limit


# Custom context manager to capture print output
@contextlib.contextmanager
def st_stdout_redirect(placeholder):
    """
    Redirects stdout to a Streamlit placeholder.
    """
    stdout_old = sys.stdout
    stringio = io.StringIO()
    sys.stdout = stringio
    try:
        yield
    finally:
        sys.stdout = stdout_old
        placeholder.text(stringio.getvalue())


# Use the user-defined limit from session_state
current_limit = st.session_state.get('university_limit', 10) # Default to 10 if not set

if st.button(f"Start Data Extraction for Thailand (Max {current_limit} Universities)", disabled=not st.session_state.get('openai_configured', False)):
    if not st.session_state.get('openai_configured', False):
        st.error("Please configure your OpenAI API Key in the sidebar first.")
    else:
        st.info("Starting data extraction... This might take a while.")
        progress_text = st.empty()
        
        all_extracted_university_data_for_streamlit = []
        final_university_data = []
        processed_count = 0

        # Create a placeholder for live output
        output_placeholder = st.empty()

        try:
            # Step 1: Initial Data Extraction from Wikipedia (Thailand only, limited by user input)
            with st.spinner(f"Extracting initial university list from Wikipedia (first {current_limit} found)..."):
                with st_stdout_redirect(output_placeholder):
                    for country in ["Thailand"]: # Only process Thailand as per requirement
                        wikipedia_url = f"https://en.wikipedia.org/wiki/List_of_universities_in_{country}"
                        st.text(f"\nProcessing {country} from {wikipedia_url}")
                        
                        country_universities_data = extract_university_tables_from_url(wikipedia_url, country)
                        
                        # Limit to user-defined number of universities from Wikipedia for further processing
                        all_extracted_university_data_for_streamlit.extend(country_universities_data[:current_limit])
                
                st.success(f"Finished initial data extraction. Total universities found (limited to {current_limit}): {len(all_extracted_university_data_for_streamlit)}")

            # Step 2: Detailed Processing with OpenAI and Google Searches
            st.info("Starting detailed processing for each university...")
            
            total_unis_to_process = len(all_extracted_university_data_for_streamlit)
            progress_bar = st.progress(0)

            for i, uni_info in enumerate(all_extracted_university_data_for_streamlit):
                if processed_count >= current_limit: # Enforce overall limit by user
                    st.warning(f"Reached maximum of {current_limit} universities for detailed processing. Stopping.")
                    break

                university_name = uni_info.get('University')
                if not university_name:
                    st.warning(f"Skipping a record due to missing 'University' name: {uni_info}")
                    continue

                progress_text.text(f"Processing ({i+1}/{total_unis_to_process}): {university_name}")
                progress_bar.progress((i + 1) / total_unis_to_process)

                with st_stdout_redirect(output_placeholder):
                    st.text(f"\n--- Processing: {university_name} ---")
                    
                    website = uni_info.get('Website', 'N/A')
                    country = uni_info.get('Country', 'N/A')
                    region = ASEAN_REGIONS.get(country, 'Unknown')

                    has_agriculture = check_with_openai(university_name)
                    
                    if has_agriculture:
                        st.text(f"  -> Has Agriculture Department: Yes")
                        has_tto = check_with_openai_TTO(university_name)
                        tto_page_url = "N/A"
                        if has_tto:
                            tto_page_url = get_tto_page_url(university_name, website)
                            st.text(f"  -> Has TTO: Yes, TTO Page URL: {tto_page_url}")
                        else:
                            st.text(f"  -> Has TTO: No")

                        incubation_record = get_incubation_record(university_name, website)
                        st.text(f"  -> Incubation Record: {incubation_record}")
                        
                        linkedin_search_url = find_university_linkedin(university_name)
                        st.text(f"  -> Apollo/LinkedIn Search URL: {linkedin_search_url}")

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
                        processed_count += 1
                    else:
                        st.text(f"  -> Has Agriculture Department: No (Skipping detailed processing)")
            
            progress_bar.empty()
            progress_text.empty()

            if final_university_data:
                df = pd.DataFrame(final_university_data)
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False, encoding='utf-8')
                csv_bytes = csv_buffer.getvalue().encode('utf-8')

                st.success(f"Successfully processed {len(final_university_data)} universities.")
                st.download_button(
                    label="Download University Data CSV",
                    data=csv_bytes,
                    file_name=f"asean_universities_data_thailand_{len(final_university_data)}_limited.csv", # Dynamic filename
                    mime="text/csv",
                )
                st.dataframe(df) # Display the DataFrame in the app
            else:
                st.warning("No universities with agriculture departments were found to generate the CSV.")

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
            st.exception(e) # Show full traceback
