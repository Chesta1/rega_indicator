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
        return pd.read_excel("city_district_mapping.xlsx", engine='openpyxl')
    except FileNotFoundError:
        # If the file is not found, show an error message
        st.error("city_district_mapping.xlsx file not found. Please place the Excel file in the same directory as this application.")
        
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
    """Fetch data from the API with exact payload format and retry on 429 errors"""
    url = "https://rentalrei.rega.gov.sa/RegaIndicatorsAPIs/api/IndicatorEjar/GetDetailsV2"
    
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
    
    retries = 3
    backoff = 10  # initial delay in seconds
    
    for attempt in range(1, retries + 1):
        try:
            response = requests.post(
                url,
                json=payload,
                headers=get_headers(),
                verify=False,
                timeout=30
            )
            
            # Check for rate limiting (status code 429)
            if response.status_code == 429:
                debug_container.write(f"Rate limit reached (attempt {attempt}/{retries}). Waiting {backoff} seconds...")
                time.sleep(backoff)
                backoff *= 2  # exponential backoff
                continue
                
            response.raise_for_status()
            debug_container.empty()
            
            try:
                json_data = response.json()
                if isinstance(json_data, list):
                    return json_data
                elif isinstance(json_data, dict):
                    if 'data' in json_data and isinstance(json_data['data'], list):
                        return json_data['data']
                    return [json_data]
                else:
                    st.warning(f"Unexpected response format for {total_rooms} rooms")
                    return []
            except json.JSONDecodeError as je:
                st.error(f"JSON parse error for {total_rooms} rooms: {str(je)}")
                return []
                
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                debug_container.write(f"Rate limit reached (attempt {attempt}/{retries}). Waiting {backoff} seconds...")
                time.sleep(backoff)
                backoff *= 2  # exponential backoff
            else:
                st.error(f"HTTP error fetching data for {total_rooms} rooms: {str(e)}")
                return []
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching data for {total_rooms} rooms: {str(e)}")
            return []
    
    st.error(f"Failed to fetch data for {total_rooms} rooms after {retries} attempts.")
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

def display_results(results, selected_district=None, date_period=None):
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
        
        # Add the district name and date period if provided and not already in the data
        if selected_district and 'district_name' not in df.columns:
            df['district_name'] = selected_district
        
        if date_period and 'date_period' not in df.columns:
            df['date_period'] = date_period
        
        # Add room type description for clarity if room_type exists
        if 'room_type' in df.columns:
            # First define the apartment room type mapping
            apartment_room_type_map = {
                0: "All Rooms",
                2: "2 Bedrooms", 
                3: "3 Bedrooms",
                4: "4 Bedrooms"
            }
            
            # Define specific descriptions for non-apartment property types
            unit_name_descriptions = {
                "duplex": "Duplex Unit",
                "floor": "Full Floor",
                "office_space": "Office Space",
                "shop": "Retail Shop",
                "studio": "Studio Unit",
                "trade_exhibiti": "Trade Exhibition Space",
                # Add any other property types here
            }
            
            # Create a custom room_type_desc based on both room_type and unitName
            def get_room_type_desc(row):
                try:
                    # For apartments, use the bedroom-based mapping
                    if 'unitName' in row and str(row['unitName']).lower() in ['apartment', 'appartment']:
                        return apartment_room_type_map.get(row['room_type'], f"{row['room_type']} Bedrooms")
                    else:
                        # For non-apartments, use the unit name mapping
                        if 'unitName' in row:
                            return unit_name_descriptions.get(row['unitName'], f"Other: {row['unitName']}")
                        else:
                            return f"{row['room_type']} Rooms"
                except Exception as e:
                    return f"{row['room_type']} Rooms"
            
            # Apply the custom mapping
            df['room_type_desc'] = df.apply(get_room_type_desc, axis=1)
            
            # First, show the original dataset size
            original_size = len(df)
            st.text(f"Original dataset: {original_size} rows")
            
            # Define room type columns
            room_cols = ['room_type', 'room_type_desc']
            
            # This approach will work with unhashable types:
            # 1. For apartments, convert to string representation for deduplication
            # 2. For non-apartments, just keep one row per unitName
            
            if 'unitName' in df.columns:
                try:
                    # Create a mask for apartments - safely handle unitName
                    apartment_mask = df['unitName'].astype(str).str.lower().isin(['apartment', 'appartment'])
                    
                    # Process apartments - need to handle possible unhashable types
                    apartments_df = df[apartment_mask].copy()
                    if not apartments_df.empty:
                        # Generate a signature for each row to identify duplicates
                        try:
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
                            
                            apartments_dedup = pd.DataFrame(apartments_dedup_rows)
                            if '_signature' in apartments_dedup.columns:
                                apartments_dedup = apartments_dedup.drop(columns=['_signature'])
                        except Exception as e:
                            st.warning(f"Error deduplicating apartments: {str(e)}")
                            apartments_dedup = apartments_df
                    else:
                        apartments_dedup = apartments_df
                    
                    # Process non-apartments - simpler, just keep one row per unitName
                    non_apartments_df = df[~apartment_mask]
                    if not non_apartments_df.empty:
                        # Keep just one record per unitName
                        try:
                            unique_unit_names = non_apartments_df['unitName'].unique()
                            non_apartments_dedup_rows = []
                            for unit_name in unique_unit_names:
                                matching_rows = non_apartments_df[non_apartments_df['unitName'] == unit_name]
                                if not matching_rows.empty:
                                    non_apartments_dedup_rows.append(matching_rows.iloc[0])
                                    
                            non_apartments_dedup = pd.DataFrame(non_apartments_dedup_rows)
                        except Exception as e:
                            st.warning(f"Error deduplicating non-apartments: {str(e)}")
                            non_apartments_dedup = non_apartments_df
                    else:
                        non_apartments_dedup = non_apartments_df
                    
                    # Combine the results
                    try:
                        df = pd.concat([apartments_dedup, non_apartments_dedup]).reset_index(drop=True)
                    except Exception as e:
                        st.warning(f"Error combining deduped data: {str(e)}")
                        df = df.copy()  # Keep original if concat fails
                except Exception as e:
                    st.warning(f"Error during deduplication: {str(e)}")
                    # Keep the original dataframe if any error occurs during deduplication
                    df = df.copy()
                
                # Show stats about duplicates removed
                final_size = len(df)
                duplicate_count = original_size - final_size
                
                if duplicate_count > 0:
                    st.info(f"Removed {duplicate_count} duplicate records ({duplicate_count / original_size * 100:.1f}% of data)")
                    st.text(f"Final dataset: {final_size} rows")
            
            # Move room_type_desc to the front and exclude room_type from display
            if not df.empty:
                cols = df.columns.tolist()
                
                # Determine which columns to place at the front
                front_cols = []
                if 'district_name' in cols:
                    front_cols.append('district_name')
                    cols.remove('district_name')
                
                if 'date_period' in cols:
                    front_cols.append('date_period')
                    cols.remove('date_period')
                
                if 'room_type_desc' in cols:
                    front_cols.append('room_type_desc')
                    cols.remove('room_type_desc')
                
                # Combine the front columns with the rest
                cols = front_cols + [col for col in cols if col in df.columns]
                
                # Remove room_type from the columns to display
                if 'room_type' in cols:
                    cols.remove('room_type')
                
                # Ensure all columns exist in the dataframe
                valid_cols = [col for col in cols if col in df.columns]
                df = df[valid_cols]
        
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
        if 'district_name' in original_df:
            flattened_df['district_name'] = original_df['district_name']
            
        if 'date_period' in original_df:
            flattened_df['date_period'] = original_df['date_period']
            
        if 'room_type_desc' in original_df:
            flattened_df['room_type_desc'] = original_df['room_type_desc']
            
        if 'unitName' in original_df:
            flattened_df['unitName'] = original_df['unitName']
            
        # Keep room_type for internal processing but don't include in final display
        if 'room_type' in original_df:
            room_type_temp = original_df['room_type']  # Store for processing if needed
            
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
            
        with col2:
            flattened_csv = flattened_df.to_csv(index=False)
            st.download_button(
                label="Download Flattened CSV",
                data=flattened_csv,
                file_name="flattened_api_data.csv",
                mime="text/csv"
            )
            
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
            # City dropdown
            city_options = cities_df['City'].unique().tolist()
            selected_city = st.selectbox("Select City", city_options)
            
            # Get the city_id for the selected city
            city_id = int(cities_df[cities_df['City'] == selected_city]['City_ID'].iloc[0])
            
            # Filter districts for the selected city
            city_districts = mapping_data[mapping_data['City'] == selected_city]
          
            # Add "All" option to district dropdown
            district_list = city_districts['District_Name'].tolist()
            district_options = ["All"] + district_list
            
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
            
            # If a specific district is selected, get its ID; otherwise, note that "All" is selected
            if selected_district != "All":
                trigger_points = int(city_districts[city_districts['District_Name'] == selected_district]['District_ID'].iloc[0])
                st.info(f"City ID: {city_id}, District ID: {trigger_points}")
            else:
                trigger_points = None  # No specific district ID when 'All' is selected
                st.info(f"City ID: {city_id}, All districts selected")
            
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
                
            # Configure delay between district requests when "All" is selected
            if selected_district == "All":
                delay_seconds = st.slider("Delay between district requests (seconds)", 
                                         min_value=5, max_value=30, value=15, 
                                         help="Longer delays reduce the chance of API rate limiting")
        
        with col2:
            # Select between custom date range and quarterly date range
            date_mode = st.radio("Choose Date Input Mode", ["Custom Date Range", "Quarterly Date Range"])
            
            # Initialize date_period variable for display
            date_period = None
            
            if date_mode == "Custom Date Range":
                # Use default custom dates
                default_start = datetime.strptime("2024-01-31", "%Y-%m-%d")
                default_end = datetime.strptime("2024-03-31", "%Y-%m-%d")
                start_date = st.date_input("Start Date", value=default_start)
                end_date = st.date_input("End Date", value=default_end)
                
                # Create a custom date range label for display
                date_period = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
            else:
                # Quarterly Date Range input
                quarter = st.selectbox("Select Quarter", ["Q1", "Q2", "Q3", "Q4"])
                year = st.number_input("Year", min_value=2000, value=2024, step=1)
                
                # Set date period label
                date_period = f"{quarter} {year}"
                
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
        
        with st.expander("Show Request Information", expanded=False):
            st.write("City:", selected_city)
            st.write("City ID:", city_id)
            st.write("District:", selected_district)
            if selected_district != "All":
                st.write("District ID:", trigger_points)
            st.write("Start Date:", start_datetime)
            st.write("End Date:", end_datetime)
            st.write("Date Period:", date_period)
            st.write("Room Types:", room_types)
            st.write("Headers:", get_headers())
        
        if st.button("Fetch Data for All Selected Room Types", type="primary") and room_types:
            with st.spinner("Fetching data..."):
                results = []
                
                if selected_district == "All":
                    # If "All" is selected, iterate through all districts in the city
                    total_districts = len(city_districts)
                    st.warning(f"Fetching data for all {total_districts} districts. This may take some time.")
                    
                    # Create progress bars
                    district_progress = st.progress(0)
                    district_status = st.empty()
                    room_progress = st.empty()
                    room_progress_bar = st.progress(0)
                    
                    # Iterate through each district
                    for idx, (_, district_row) in enumerate(city_districts.iterrows()):
                        current_district_name = district_row["District_Name"]
                        current_district_id = int(district_row["District_ID"])
                        
                        # Update district progress
                        district_progress.progress((idx) / total_districts)
                        district_status.info(f"Processing district {idx+1}/{total_districts}: {current_district_name} (ID: {current_district_id})")
                        
                        # Fetch data for all room types in this district
                        district_results = fetch_all_room_types(
                            city_id,
                            start_datetime,
                            end_datetime,
                            room_types,
                            current_district_id,
                            room_progress_bar
                        )
                        
                        # Add the current district name to each record
                        for record in district_results:
                            record['district_name'] = current_district_name
                            record['district_id'] = current_district_id
                        
                        results.extend(district_results)
                        
                        # Report success or failure for this district
                        if district_results:
                            st.success(f"Retrieved {len(district_results)} records for {current_district_name}")
                        else:
                            st.error(f"Failed to retrieve data for {current_district_name}")
                        
                        # Wait before processing the next district (except for the last one)
                        if idx < total_districts - 1:
                            wait_msg = f"Waiting {delay_seconds} seconds before processing the next district..."
                            with st.spinner(wait_msg):
                                time.sleep(delay_seconds)
                    
                    # Complete the district progress bar
                    district_progress.progress(1.0)
                    district_status.success(f"Completed processing all {total_districts} districts!")
                    
                else:
                    # Single district case
                    progress_bar = st.empty()
                    progress_bar_placeholder = st.progress(0)
                    
                    results = fetch_all_room_types(
                        city_id,
                        start_datetime,
                        end_datetime,
                        room_types,
                        trigger_points,
                        progress_bar
                    )
                    
                    progress_bar.empty()
                    progress_bar_placeholder.empty()
                
                # Process and display results
                if results:
                    st.success(f"Successfully fetched {len(results)} total records")
                    
                    with st.expander("Raw Combined API Response", expanded=False):
                        st.json(results[:5])  # Show only first 5 records to avoid UI slowdown
                    
                    # Pass the district information to display_results
                    display_results(
                        results, 
                        None if selected_district == "All" else selected_district,  # Don't override district names when "All" is selected
                        date_period
                    )
                else:
                    st.error("Failed to fetch any data. Please try again with different parameters or check network connectivity.")
    
    with tab2:
        st.header("About this application")
        st.write("""
        This application extracts rental data for multiple room types from the Rega API service.
        
        ### Features:
        - Extract data for multiple room types in a single operation
        - Fetch data for either a single district or all districts in a city
        - Combine results into a single downloadable dataset
        - Support for custom date ranges or quarterly periods
        - User-friendly city and district selection
        
        ### Room Types:
        - All Rooms (0 rooms)
        - 2 Bedrooms
        - 3 Bedrooms
        - 4 Bedrooms
        
        ### Tips:
        - If requests fail, try with fewer room types at once
        - When fetching all districts, use a longer delay (15+ seconds) to avoid API rate limits
        - The "All districts" option will take significantly longer to complete
        """)
        
        with st.expander("View City-District Mapping Reference", expanded=False):
            display_city_district_mapping(mapping_data)
            
            st.info("The application loads city and district data from 'city_district_mapping.xlsx'. Make sure this file is in the same directory as the application to update the available cities and districts.")

if __name__ == "__main__":
    main()