## Download Processed Data

Please download the processed data from this link: [Download Processed Data](<https://drive.google.com/drive/folders/1TTSAN66Rwu1KIyvkepZ3VT3VWZ_0avJP?usp=sharing>)

## Process Your Own Data (Optional)

If you want to generate the processed datasets locally:

- Place raw CSVs at:
  - data/original/CICIDS2017/CICIDS2017.csv
  - data/original/CICDDoS2019/CICDDoS2019.csv

- Run with the helper script:
  ```bash
  cd scripts
  bash data_preprocess.sh
  ```
  Or from the repo root:
  ```bash
  python3 data_preprocess.py -d 2017
  python3 data_preprocess.py -d 2019
  ```

Outputs will be saved under data/processed/.
