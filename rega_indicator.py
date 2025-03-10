import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import json
import urllib3
import time

# Disable SSL verification warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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
    if not results:
        st.warning("No data available to display")
        return
    
    try:
        # First, normalize complex structures in the results
        # This is critical for handling lists and dictionaries that cause unhashable type errors
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
        
        # Check if room_type exists in the DataFrame
        if 'room_type' in df.columns:
            # Add room type description for clarity
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
            final_df = pd.concat([apartments_dedup, non_apartments_dedup]).reset_index(drop=True)
            
            # Show stats about duplicates removed
            final_size = len(final_df)
            duplicate_count = original_size - final_size
            
            if duplicate_count > 0:
                st.info(f"Removed {duplicate_count} duplicate records ({duplicate_count / original_size * 100:.1f}% of data)")
                st.text(f"Final dataset: {final_size} rows")
            
            # Move room type columns to the front for better visibility
            if not final_df.empty:
                cols = final_df.columns.tolist()
                room_cols_present = [col for col in room_cols if col in cols]
                other_cols = [col for col in cols if col not in room_cols]
                final_df = final_df[room_cols_present + other_cols]
        else:
            st.warning("Room type information not found in the response data")
            final_df = df
        
        st.subheader("Combined Data (Duplicates Removed)")
        st.dataframe(final_df)
        
        # Download options
        col1, col2 = st.columns(2)
        with col1:
            csv = final_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="multi_room_api_data.csv",
                mime="text/csv"
            )
        with col2:
            # Convert dataframe back to JSON for download
            json_str = final_df.to_json(orient="records", indent=2)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name="multi_room_api_data.json",
                mime="application/json"
            )
        
        # Display basic stats by room type if possible
        if 'room_type_desc' in final_df.columns and 'averagePrice' in final_df.columns:
            st.subheader("Summary by Room Type")
            summary = final_df.groupby('room_type_desc')['averagePrice'].agg(['mean', 'min', 'max', 'count']).reset_index()
            summary.columns = ['Room Type', 'Average Price', 'Min Price', 'Max Price', 'Count']
            st.dataframe(summary)
    except Exception as e:
        st.error(f"Error processing results: {str(e)}")
        # Print more detailed error information
        import traceback
        st.error(traceback.format_exc())
        st.json(results)  # Display raw JSON as fallback

def main():
    st.set_page_config(page_title="Multiple Room Types Data Extractor", layout="wide")
    st.title("📊 Multiple Room Types Data Extractor")
    
    # Create tabs for better organization
    tab1, tab2 = st.tabs(["Extract Data", "About"])
    
    with tab1:
        # Input section for basic API parameters
        col1, col2 = st.columns(2)
        
        with col1:
            city_id = st.number_input("City ID", value=21282)
            trigger_points = st.text_input("Trigger Points", value="2204")
            
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
        
        ### Room Types:
        - Studio (0 rooms)
        - 2 Bedrooms
        - 3 Bedrooms
        - 4 Bedrooms
        
        ### Tips:
        - If requests fail, try with fewer room types at once
        - The API might have rate limits, so allow some time between extractions
        """)

if __name__ == "__main__":
    main()

