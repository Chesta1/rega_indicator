import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import json
import urllib3
import time
import io

# Disable SSL verification warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load city and district data from external CSV file
@st.cache_data
def load_city_district_data():
    """
    Load city and district mapping data from the external CSV file
    If the file doesn't exist, show an error and use a fallback sample
    """
    try:
        # Try to load the external CSV file
        return pd.read_excel("city_district_mapping.xlsx",engine='openpyxl')
    except FileNotFoundError:
        # If the file is not found, show an error message
        st.error("city_district_mapping.csv file not found. Please place the CSV file in the same directory as this application.")
        
        # Provide a fallback sample dataset
        data = {
            "City": ["Riyadh"],
            "City_ID": [21282],
            "District_Name": ["Al Qirawan"],
            "District_ID": [2204]
        }
        return pd.DataFrame(data)

def display_city_district_mapping(mapping_data):
    """Display the current city-district mapping data"""
    st.subheader("City-District Mapping Reference")
    
    # Display the current mapping data
    st.dataframe(mapping_data)
    
    # Provide a download of the current mapping
    st.download_button(
        label="Download Current Mapping",
        data=mapping_data.to_csv(index=False),
        file_name="city_district_mapping.csv",
        mime="text/csv"
    )

def get_headers():
    """Get headers exactly matching the browser request"""
    return {
        'accept': 'application/json, text/plain, */*',
        'accept-encoding': 'gzip, deflate, br, zstd',
        'accept-language': 'ar',
        'authorization': 'Bearer null',
        'content-type': 'application/json',
        'origin': 'https://rentalrei.rega.gov.sa',
        'referer': 'https://rentalrei.rega.gov.sa/indicatorejar',
        'sec-ch-ua': '"Not(A:Brand";v="99", "Google Chrome";v="133", "Chromium";v="133"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36'
    }

def fetch_api_data(city_id, start_date, end_date, total_rooms, trigger_points):
    """Fetch data from the API with exact payload format"""
    url = "https://rentalrei.rega.gov.sa/RegaIndicatorsAPIs/api/IndicatorEjar/GetDetailsV2"

    # Format dates exactly as expected by the API
    payload = {
        "cityId": city_id,
        "end_date": end_date.strftime("%Y-%m-%dT18:30:00.000Z"),
        "end_date2": end_date.strftime("%a %b %d %Y 00:00:00 GMT+0530 (India Standard Time)"),
        "strt_date": start_date.strftime("%Y-%m-%dT18:30:00.000Z"),
        "strt_date2": start_date.strftime("%a %b %d %Y 00:00:00 GMT+0530 (India Standard Time)"),
        "totalRooms": total_rooms,
        "trigger_Points": str(trigger_points)
    }

    debug_container = st.empty()
    with st.container():
        debug_container.write(f"Fetching data for {total_rooms} rooms...")

    try:
        response = requests.post(
            url,
            json=payload,
            headers=get_headers(),
            verify=False,
            timeout=30
        )

        # Only show detailed debug info when expanded
        with st.expander(f"Debug for {total_rooms} rooms request", expanded=False):
            st.write("Request Payload:", payload)
            st.write("Response Status Code:", response.status_code)
            st.write("Response Headers:", dict(response.headers))
            st.write("First 500 chars of Response:", response.text[:500] + "..." if len(response.text) > 500 else response.text)

        response.raise_for_status()
        debug_container.empty()
        
        # Safely parse JSON response
        try:
            json_data = response.json()
            # Add room_type information to the response at the JSON level
            if isinstance(json_data, list):
                return json_data
            elif isinstance(json_data, dict):
                # If it's a dict with 'data' key, we want to extract that
                if 'data' in json_data and isinstance(json_data['data'], list):
                    return json_data['data']
                return [json_data]  # Return as single-item list
            else:
                st.warning(f"Unexpected response format for {total_rooms} rooms")
                return []
        except json.JSONDecodeError as je:
            st.error(f"JSON parse error for {total_rooms} rooms: {str(je)}")
            return []
            
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data for {total_rooms} rooms: {str(e)}")
        return []

def fetch_all_room_types(city_id, start_date, end_date, room_types, trigger_points, progress_bar):
    """Fetch data for all specified room types and combine results"""
    all_results = []
    
    for i, room_type in enumerate(room_types):
        progress_text = f"Fetching data for {room_type} rooms ({i+1}/{len(room_types)})..."
        progress_bar.text(progress_text)
        
        result = fetch_api_data(
            city_id,
            start_date,
            end_date,
            room_type,
            trigger_points
        )
        
        if result:
            # Add room type information to each record by creating a new list with modified copies
            modified_result = []
            for record in result:
                # Create a copy of the record to avoid modifying the original
                record_copy = record.copy() if isinstance(record, dict) else {}
                record_copy['room_type'] = room_type
                modified_result.append(record_copy)
            
            all_results.extend(modified_result)
        
        # Update progress
        progress_bar.progress((i + 1) / len(room_types))
        
        # Small delay to avoid overwhelming the API
        time.sleep(1)
    
    return all_results


def display_results(results):
    """Display and offer download options for the results"""
    import pandas as pd
    import numpy as np
    import json
    import ast
    
    if not results:
        st.warning("No data available to display")
        return
    
    try:
        # First, normalize complex structures in the results
        normalized_results = []
        
        for record in results:
            # Create a flattened copy of the record
            flat_record = {}
            
            # Process all fields, converting unhashable types to strings
            for key, value in record.items():
                if isinstance(value, (list, dict)):
                    # Convert complex types to string representation
                    flat_record[key] = str(value)
                else:
                    flat_record[key] = value
            
            normalized_results.append(flat_record)
        
        # Convert to DataFrame with error handling
        df = pd.json_normalize(normalized_results)
        
        # Add room type description for clarity if room_type exists
        if 'room_type' in df.columns:
            room_type_map = {
                0: "All Rooms",
                2: "2 Bedrooms", 
                3: "3 Bedrooms",
                4: "4 Bedrooms"
            }
            df['room_type_desc'] = df['room_type'].map(lambda x: room_type_map.get(x, f"{x} Bedrooms"))
            
            # First, show the original dataset size
            original_size = len(df)
            st.text(f"Original dataset: {original_size} rows")
            
            # Define room type columns
            room_cols = ['room_type', 'room_type_desc']
            
            # This approach will work with unhashable types:
            # 1. For apartments, convert to string representation for deduplication
            # 2. For non-apartments, just keep one row per unitName
            
            # Create a mask for apartments
            apartment_mask = df['unitName'].str.lower().isin(['apartment', 'appartment'])
            
            # Process apartments - need to handle possible unhashable types
            apartments_df = df[apartment_mask].copy()
            if not apartments_df.empty:
                # Generate a signature for each row to identify duplicates
                apartments_df['_signature'] = apartments_df.apply(
                    lambda row: '|'.join(str(row[col]) for col in apartments_df.columns 
                                         if col not in ['_signature'] + room_cols),
                    axis=1
                )
                
                # Now we can safely deduplicate based on this signature and room_type
                apartments_dedup_rows = []
                for signature, group in apartments_df.groupby('_signature'):
                    for room_type in group['room_type'].unique():
                        matching_rows = group[group['room_type'] == room_type]
                        if not matching_rows.empty:
                            apartments_dedup_rows.append(matching_rows.iloc[0])
                
                apartments_dedup = pd.DataFrame(apartments_dedup_rows).drop(columns=['_signature'])
            else:
                apartments_dedup = apartments_df
            
            # Process non-apartments - simpler, just keep one row per unitName
            non_apartments_df = df[~apartment_mask]
            if not non_apartments_df.empty:
                # Keep just one record per unitName
                unique_unit_names = non_apartments_df['unitName'].unique()
                non_apartments_dedup_rows = []
                for unit_name in unique_unit_names:
                    matching_rows = non_apartments_df[non_apartments_df['unitName'] == unit_name]
                    if not matching_rows.empty:
                        non_apartments_dedup_rows.append(matching_rows.iloc[0])
                        
                non_apartments_dedup = pd.DataFrame(non_apartments_dedup_rows)
            else:
                non_apartments_dedup = non_apartments_df
            
            # Combine the results
            df = pd.concat([apartments_dedup, non_apartments_dedup]).reset_index(drop=True)
            
            # Show stats about duplicates removed
            final_size = len(df)
            duplicate_count = original_size - final_size
            
            if duplicate_count > 0:
                st.info(f"Removed {duplicate_count} duplicate records ({duplicate_count / original_size * 100:.1f}% of data)")
                st.text(f"Final dataset: {final_size} rows")
            
            # Move room type columns to the front for better visibility
            if not df.empty:
                cols = df.columns.tolist()
                room_cols_present = [col for col in room_cols if col in cols]
                other_cols = [col for col in cols if col not in room_cols]
                df = df[room_cols_present + other_cols]
        
        # This is our original dataframe (after deduplication)
        original_df = df.copy()
        
        # Create a function to extract and flatten the JSON data
        def extract_range_data(json_str, range_key):
            """Extract data from a specific range in the JSON string"""
            try:
                # Handle non-string inputs
                if not isinstance(json_str, str):
                    return None, None, None
                
                # Handle empty strings
                if not json_str or json_str.strip() == '':
                    return None, None, None
                
                # Convert single quotes to double quotes for JSON parsing
                json_str = json_str.replace("'", '"')
                
                # Try to parse the JSON
                try:
                    data = json.loads(json_str)
                except:
                    try:
                        data = ast.literal_eval(json_str)
                    except:
                        return None, None, None
                
                # Check if the range_key exists in the data
                if not data or range_key not in data:
                    return None, None, None
                
                range_data = data[range_key]
                
                # Handle case where range_data is a list with one element
                if isinstance(range_data, list) and len(range_data) > 0:
                    range_data = range_data[0]
                
                # Handle case where range_data is None or not a dict
                if not range_data or not isinstance(range_data, dict):
                    return None, None, None
                    
                # Extract the values we're interested in
                sum_deals = range_data.get('sumDeals', None)
                avg_max = range_data.get('avgMax', None)
                avg_min = range_data.get('avgMin', None)
                
                return sum_deals, avg_max, avg_min
            except Exception as e:
                # Just return None values instead of raising an error
                return None, None, None
        
        # Now create a separate, simplified and flattened dataframe
        st.subheader("Creating Flattened Data")
        
        # Start with a fresh DataFrame for the flattened data
        flattened_df = pd.DataFrame()
        
        # Add the core identifying columns first
        if 'room_type' in original_df:
            flattened_df['room_type'] = original_df['room_type']
        if 'room_type_desc' in original_df:
            flattened_df['room_type_desc'] = original_df['room_type_desc']
        if 'unitName' in original_df:
            flattened_df['unitName'] = original_df['unitName']
            
        # Add important numeric columns
        for col in ['sumRent', 'sumDeals', 'avg', 'avgMax', 'avgMin']:
            if col in original_df:
                flattened_df[col] = original_df[col]
        
        # Process unitRanges
        if 'unitRanges' in original_df.columns:
            for range_num in ['range1', 'range2', 'range3']:
                # Add columns for each extracted value
                flattened_df[f'unitRanges_{range_num}_sumDeals'] = original_df['unitRanges'].apply(
                    lambda x: extract_range_data(x, range_num)[0]
                )
                flattened_df[f'unitRanges_{range_num}_avgMax'] = original_df['unitRanges'].apply(
                    lambda x: extract_range_data(x, range_num)[1]
                )
                flattened_df[f'unitRanges_{range_num}_avgMin'] = original_df['unitRanges'].apply(
                    lambda x: extract_range_data(x, range_num)[2]
                )
        
        # Process unitsRanges
        if 'unitsRanges' in original_df.columns:
            for range_num in ['range1', 'range2', 'range3']:
                # Add columns for each extracted value
                flattened_df[f'unitsRanges_{range_num}_sumDeals'] = original_df['unitsRanges'].apply(
                    lambda x: extract_range_data(x, range_num)[0]
                )
                flattened_df[f'unitsRanges_{range_num}_avgMax'] = original_df['unitsRanges'].apply(
                    lambda x: extract_range_data(x, range_num)[1]
                )
                flattened_df[f'unitsRanges_{range_num}_avgMin'] = original_df['unitsRanges'].apply(
                    lambda x: extract_range_data(x, range_num)[2]
                )
        
        # Add any remaining columns from the original dataframe that weren't already added
        # and aren't the JSON columns we've already processed
        excluded_cols = ['unitRanges', 'unitsRanges']
        for col in original_df.columns:
            if col not in flattened_df.columns and col not in excluded_cols:
                flattened_df[col] = original_df[col]
        
        # Show both versions with tabs
        st.subheader("Data Display")
        tab1, tab2 = st.tabs(["Original Data", "Flattened Data"])
        
        with tab1:
            st.dataframe(original_df)
            
        with tab2:
            st.dataframe(flattened_df)
        
        # Download options
        col1, col2 = st.columns(2)
        
        with col1:
            csv = original_df.to_csv(index=False)
            st.download_button(
                label="Download Original CSV",
                data=csv,
                file_name="original_api_data.csv",
                mime="text/csv"
            )
            
            # Convert dataframe back to JSON for download
            class NumpyJSONEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, (np.integer, np.int64)):
                        return int(obj)
                    elif isinstance(obj, (np.floating, np.float64)):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return super(NumpyJSONEncoder, self).default(obj)
                    
            # json_str = json.dumps(original_df.to_dict(orient="records"), indent=2, cls=NumpyJSONEncoder)
            # st.download_button(
            #     label="Download Original JSON",
            #     data=json_str,
            #     file_name="original_api_data.json",
            #     mime="application/json"
            # )
            
        with col2:
            flattened_csv = flattened_df.to_csv(index=False)
            st.download_button(
                label="Download Flattened CSV",
                data=flattened_csv,
                file_name="flattened_api_data.csv",
                mime="text/csv"
            )
            
            # # Convert flattened dataframe to JSON for download
            # flattened_json_str = json.dumps(flattened_df.to_dict(orient="records"), indent=2, cls=NumpyJSONEncoder)
            # st.download_button(
            #     label="Download Flattened JSON",
            #     data=flattened_json_str,
            #     file_name="flattened_api_data.json",
            #     mime="application/json"
            # )
        
        # # Display basic stats by room type if possible
        # if 'room_type_desc' in original_df.columns and 'averagePrice' in original_df.columns:
        #     st.subheader("Summary by Room Type")
        #     summary = original_df.groupby('room_type_desc')['averagePrice'].agg(['mean', 'min', 'max', 'count']).reset_index()
        #     summary.columns = ['Room Type', 'Average Price', 'Min Price', 'Max Price', 'Count']
        #     st.dataframe(summary)
            
    except Exception as e:
        st.error(f"Error processing results: {str(e)}")
        # Print more detailed error information
        import traceback
        st.error(traceback.format_exc())
        st.json(results)  # Display raw JSON as fallback

def main():
    st.set_page_config(page_title="Rental Real Estate Indicator Data", layout="wide")
    st.title("📊 Rental Real Estate Indicator Data")
    
    # Load city-district mapping data
    mapping_data = load_city_district_data()
    
    # Get unique cities with their IDs

    cities_df = mapping_data[['City', 'City_ID']].drop_duplicates()
    
    # Create tabs for better organization
    tab1, tab2 = st.tabs(["Extract Data", "About"])
    
    with tab1:
        # Input section for basic API parameters
        col1, col2 = st.columns(2)
        
        with col1:
            # City dropdown (replaces city_id number input)
            city_options = cities_df['City'].unique().tolist()
            selected_city = st.selectbox("Select City", city_options)
            
            # Get the city_id for the selected city (convert from numpy int64 to Python int)
            city_id = int(cities_df[cities_df['City'] == selected_city]['City_ID'].iloc[0])
            
            # Filter districts for the selected city
            city_districts = mapping_data[mapping_data['City'] == selected_city]
          
            # District dropdown (replaces trigger_points text input)
            district_options = city_districts['District_Name'].tolist()
            # Inject custom CSS to support Arabic fonts
            st.markdown(
                """
                <style>
                /* This targets Streamlit widgets; you might need to adjust the selector after inspecting the rendered HTML */
                .stSelectbox, .stSelectbox * {
                    font-family: 'Noto Naskh Arabic', Arial, sans-serif;
                }
                </style>
                """,
                unsafe_allow_html=True
                )
            selected_district = st.selectbox("Select District", district_options)
            
            # Get the district_id (trigger_points) for the selected district (convert from numpy int64 to Python int)
            trigger_points = int(city_districts[city_districts['District_Name'] == selected_district]['District_ID'].iloc[0])
            
            # Show the actual numeric IDs (for transparency)
            st.info(f"City ID: {city_id}, District ID: {trigger_points}")
            
            # Room type selection (checkbox for each type)
            st.subheader("Select Room Types")
            room_types = []
            if st.checkbox("All rooms (0 rooms)", value=True):
                room_types.append(0)
            if st.checkbox("2 Bedrooms", value=True):
                room_types.append(2)
            if st.checkbox("3 Bedrooms", value=True):
                room_types.append(3)
            if st.checkbox("4 Bedrooms", value=True):
                room_types.append(4)
            
            if not room_types:
                st.warning("Please select at least one room type")
        
        with col2:
            # Select between custom date range and quarterly date range
            date_mode = st.radio("Choose Date Input Mode", ["Custom Date Range", "Quarterly Date Range"])
            
            if date_mode == "Custom Date Range":
                # Use default custom dates
                default_start = datetime.strptime("2024-01-31", "%Y-%m-%d")
                default_end = datetime.strptime("2024-03-31", "%Y-%m-%d")
                start_date = st.date_input("Start Date", value=default_start)
                end_date = st.date_input("End Date", value=default_end)
            else:
                # Quarterly Date Range input
                quarter = st.selectbox("Select Quarter", ["Q1", "Q2", "Q3", "Q4"])
                year = st.number_input("Year", min_value=2000, value=2024, step=1)
                if quarter == "Q1":
                    start_date = datetime(year, 1, 1)
                    end_date = datetime(year, 3, 31)
                elif quarter == "Q2":
                    start_date = datetime(year, 4, 1)
                    end_date = datetime(year, 6, 30)
                elif quarter == "Q3":
                    start_date = datetime(year, 7, 1)
                    end_date = datetime(year, 9, 30)
                else:  # Q4
                    start_date = datetime(year, 10, 1)
                    end_date = datetime(year, 12, 31)
        
        # Combine the date with the minimum time to form a datetime object
        start_datetime = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.min.time())
        
        with st.expander("Show Request Information"):
            st.write("City:", selected_city)
            st.write("City ID:", city_id)
            st.write("District:", selected_district)
            st.write("District ID:", trigger_points)
            st.write("Start Date:", start_datetime)
            st.write("End Date:", end_datetime)
            st.write("Room Types:", room_types)
            st.write("Headers:", get_headers())
        
        if st.button("Fetch Data for All Selected Room Types", type="primary") and room_types:
            with st.spinner("Fetching data for all room types..."):
                # Create progress bar
                progress_bar = st.empty()
                progress_bar_placeholder = st.progress(0)
                
                # Fetch data for all room types
                results = fetch_all_room_types(
                    city_id,
                    start_datetime,
                    end_datetime,
                    room_types,
                    trigger_points,
                    progress_bar
                )
                
                # Clear progress elements
                progress_bar.empty()
                progress_bar_placeholder.empty()
                
                if results:
                    st.success(f"Successfully fetched data for {len(room_types)} room types with {len(results)} total records")
                    
                    with st.expander("Raw Combined API Response"):
                        st.json(results)
                    
                    display_results(results)
                else:
                    st.error("Failed to fetch data for one or more room types")
    
    with tab2:
        st.subheader("About this application")
        st.write("""
        This application extracts rental data for multiple room types from the Rega API service.
        
        ### Features:
        - Extract data for multiple room types in a single operation
        - Combine results into a single downloadable dataset
        - Visualize summary statistics by room type
        - Support for custom date ranges or quarterly periods
        - User-friendly city and district selection
        
        ### Room Types:
        - All Rooms (0 rooms)
        - 2 Bedrooms
        - 3 Bedrooms
        - 4 Bedrooms
        
        ### Tips:
        - If requests fail, try with fewer room types at once
        - The API might have rate limits, so allow some time between extractions
        """)
        
        with st.expander("View City-District Mapping Reference", expanded=False):
            display_city_district_mapping(mapping_data)
            
            st.info("The application loads city and district data from 'city_district_mapping.csv'. Make sure this file is in the same directory as the application to update the available cities and districts.")

if __name__ == "__main__":
    main()