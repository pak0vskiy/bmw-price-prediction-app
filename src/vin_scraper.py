import requests, json, os, shutil
from time import sleep

backup_path = "vin_data_backup.json"
file_path = "vin_data.json"

# Get vins data from NHTSA vehicle API
def get_vin_data(vins_str, url="https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVINValuesBatch/"):
    post_fields = {'format': 'json', 'data': vins_str}
    r = requests.post(url, data=post_fields)
    if r.status_code == 200:
        return json.loads(r.text)  # Parse JSON response
    else:
        print(f"Error: {r.status_code}, {r.text}")
        return {}


#Create batches of 50 (max limitation of API)
for idx, i in enumerate(range(0, len(vins), 50)):
    end = i + 50
    vins_str = ";".join(vins[i:end])
    tmp = get_vin_data(vins_str=vins_str)
    
    with open(file_path, 'a') as file:  # Append mode
        #Write the batch as a dictionary
        json.dump({"batch_index": idx, "data": tmp}, file)
        file.write(',\n')  # Add a comma and newline for separation

    print(f"Batch from {str(i)} to {str(end)} successfully stored")
    sleep(1)

import pandas as pd
import ijson

file_path = "vin_data.json"

df = pd.DataFrame()


batch_counter = 0

# Stream the JSON file to process batch-by-batch
with open(file_path, "r") as file:
    for batch in ijson.items(file, "item"):
        batch_counter += 1  
        
        
        
        # Extract "Results" key from the "data" dictionary
        results = batch.get("data", {}).get("Results", [])
        
        # Debugging: Show the first few entries of "Results"
        print(f"Results size: {len(results)}")
        
        # Convert the results list into a DataFrame
        batch_df = pd.DataFrame(results)
        
        # Append the batch DataFrame to the main DataFrame
        df = pd.concat([df, batch_df], ignore_index=True)

        # Show the shape of the DataFrame
        print(f"DataFrame shape after processing batch {batch_counter}: {df.shape}")

print("Processing complete. Final DataFrame:")
print(df.info())
