#!/bin/bash

# --- Configuration ---
BASE_REMOTE_URL="https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes"

# Dynamically targets your home directory
LOCAL_BASE_DIR="$HOME/SEP_study/SGPS_data"

SATELLITES=("goes16" "goes17" "goes18" "goes19")
# Dataset types: 1-minute and 5-minute averages
AVG_TYPES=("sgps-l2-avg1m" "sgps-l2-avg5m")
END_YEAR=2026

# --- Main Script ---

echo "Starting download process for GOES SGPS L2 satellite data (1m and 5m averages)."

for SAT in "${SATELLITES[@]}"; do
    echo -e "\n=========================================="
    echo "Processing Satellite: $SAT"
    echo "=========================================="

    # Define operational start years
    case "$SAT" in
        goes16) START_YEAR=2017 ;;
        goes17) START_YEAR=2018 ;;
        goes18) START_YEAR=2022 ;;
        goes19) START_YEAR=2024 ;;
        *)      START_YEAR=2017 ;;
    esac

    for AVG_TYPE in "${AVG_TYPES[@]}"; do
        echo -e "\n--- Cadence: $AVG_TYPE ---"

        for YEAR in $(seq "$START_YEAR" "$END_YEAR"); do
            echo "  Processing Year: $YEAR"

            # Construct remote URL and local target directory
            # Example Remote: .../goes16/l2/data/sgps-l2-avg5m/2018/
            # Example Local:  .../SGPS_data/goes16/sgps-l2-avg5m/2018
            REMOTE_DIR_URL="${BASE_REMOTE_URL}/${SAT}/l2/data/${AVG_TYPE}/${YEAR}/"
            LOCAL_TARGET_DIR="${LOCAL_BASE_DIR}/${SAT}/${AVG_TYPE}/${YEAR}"

            echo "    Remote URL: $REMOTE_DIR_URL"
            echo "    Local Dir:  $LOCAL_TARGET_DIR"

            # Create target folder
            mkdir -p "$LOCAL_TARGET_DIR"

            if [ ! -d "$LOCAL_TARGET_DIR" ]; then
                echo "    ERROR: Failed to create target directory $LOCAL_TARGET_DIR. Skipping."
                continue
            fi

            echo "    Downloading .nc files (skipping unmodified files)..."
            
            # The -N flag enables timestamping to only download new or updated files
            wget -r -np -nd -N -A "*.nc" \
                 -e robots=off \
                 --reject "index.html*" \
                 -P "$LOCAL_TARGET_DIR" \
                 "$REMOTE_DIR_URL" \
                 --show-progress

            if [ $? -eq 0 ]; then
                echo "    Download process successful for $SAT/$AVG_TYPE/$YEAR."
            else
                echo "    WARNING: Download completed with errors or no files found for $SAT/$AVG_TYPE/$YEAR."
            fi
        done
    done
done

# Remove any empty subdirectories left behind
echo -e "\nCleaning up empty directories..."
find "$LOCAL_BASE_DIR" -mindepth 1 -type d -empty -delete

echo -e "\nAll downloads attempted. Script finished."