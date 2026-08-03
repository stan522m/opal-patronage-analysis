import requests
import tqdm
import os

for year in tqdm.tqdm((2020,2025), desc="Year"):
    for month in tqdm.tqdm(range(1,13), desc="Month"):
        for day in tqdm.tqdm(range(1,32), desc="Day"):
            year_identifier = f"{year}{month}" if month > 9 else f"{year}0{month}"
            month_identifier = f"{month}" if month > 9 else f"0{month}"
            day_identifier = f"{day}" if day > 9 else f"0{day}"
            file_name = f"https://tfnsw-prod-opendata-tpa.s3-ap-southeast-2.amazonaws.com/Opal_Patronage/{year}-{month_identifier}/Opal_Patronage_{year_identifier}{day_identifier}.txt"
            r = requests.get(file_name, allow_redirects=True, headers={
                'Referer': 'https://opendata.transport.nsw.gov.au/'
            })
            # print(r.status_code)
            if r.status_code != 200:
                break

            if not os.path.exists(f"data/{year}-{month_identifier}"):
                os.makedirs(f"data/{year}-{month_identifier}")
            open(f"data/{year}-{month_identifier}/Opal_Patronage_{year_identifier}{day_identifier}.txt", 'wb').write(r.content)