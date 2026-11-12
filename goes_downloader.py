import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from pathlib import Path
from tqdm import tqdm

# --- Configuration ---
BASE_REMOTE_URL = "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes"
LOCAL_BASE_DIR = Path(r"C:\Users\Will\Box\SEP_Box\SGPS_data")

SATELLITES = {
    "goes16": 2017,
    "goes17": 2018,
    "goes18": 2022,
    "goes19": 2024
}
AVG_TYPES = ["sgps-l2-avg5m", "sgps-l2-avg1m"]
END_YEAR = 2026

# --- Main Script ---
def main():
    print("Starting download process for GOES SGPS L2 satellite data...")
    session = requests.Session()

    for sat, start_year in SATELLITES.items():
        print(f"\n{'='*40}")
        print(f"Processing Satellite: {sat.upper()}")
        print(f"{'='*40}")

        for avg_type in AVG_TYPES:
            print(f"\n--- Cadence: {avg_type} ---")

            for year in range(start_year, END_YEAR + 1):
                print(f"  Processing Year: {year}")
                
                # Target local directory for the whole year
                local_target_dir = LOCAL_BASE_DIR / sat / avg_type / str(year)
                
                year_files_found = 0
                
                # Loop through months 01 to 12
                for month in range(1, 13):
                    month_str = f"{month:02d}" # Formats 1 to "01"
                    
                    # Remote URL now explicitly includes the month
                    remote_dir_url = f"{BASE_REMOTE_URL}/{sat}/l2/data/{avg_type}/{year}/{month_str}/"
                    
                    response = session.get(remote_dir_url)
                    if response.status_code != 200:
                        continue  # Skip if month directory doesn't exist
                    
                    soup = BeautifulSoup(response.text, 'html.parser')
                    nc_files = [a['href'] for a in soup.find_all('a') if a.get('href', '').endswith('.nc')]
                    
                    if not nc_files:
                        continue
                        
                    # Create the local directory only if files exist
                    local_target_dir.mkdir(parents=True, exist_ok=True)
                    year_files_found += len(nc_files)
                    
                    # Download loop with progress bar showing the current month
                    desc = f"    Month {month_str}"
                    for file_name in tqdm(nc_files, desc=desc, unit="file", leave=False):
                        file_url = urljoin(remote_dir_url, file_name)
                        local_file_path = local_target_dir / file_name
                        
                        # --- Smart Download (Replicating wget -N) ---
                        head = session.head(file_url)
                        remote_size = int(head.headers.get('Content-Length', 0))
                        
                        if local_file_path.exists() and local_file_path.stat().st_size == remote_size:
                            continue
                            
                        with session.get(file_url, stream=True) as r:
                            r.raise_for_status()
                            with open(local_file_path, 'wb') as f:
                                for chunk in r.iter_content(chunk_size=8192):
                                    f.write(chunk)
                                    
                if year_files_found == 0:
                    print(f"    No .nc files found for {year}.")
                else:
                    print(f"    Successfully synced {year_files_found} files for {year}.")

    # --- Cleanup ---
    print("\nCleaning up empty directories...")
    for dirpath, dirnames, filenames in os.walk(LOCAL_BASE_DIR, topdown=False):
        if not os.listdir(dirpath):
            os.rmdir(dirpath)
            
    print("\nAll downloads completed. Script finished.")

if __name__ == "__main__":
    main()