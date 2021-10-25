## Phrase Extraction by Moses

### File framework

```
NMT
  - en-de
    - SMT
      - data_deen
      - lm
      - phrase
        - deal_table.py
        - table_to_vocab.sh
      - working
        - train
          - model
      - Moses_main.sh
      - tok_tc_bpe.sh
      - train.sh
```

You need to deploy the code file in the framework described above.

### Run

Just run the main script: `Moses_main.sh` 

Note: Each step in the main script is executed separately, not together.

