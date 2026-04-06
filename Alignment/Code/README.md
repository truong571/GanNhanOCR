
## Quick start

```bash
pip install uv
uv sync
uv run align.py configs/sample.yml
```

The program creates a colorized Excel file and its CSV version while outputing the overall information in the terminal as below:

```
Alignment Results:
Total characters processed: 1238
Total matches: 887
Overall accuracy: 71.65%
Mismatch rate: 28.35%
Alignment completed successfully!
```

## Notes

- `dict` directory provides Vietnamese-Nom dictionary and Top-k Similarity List as CSV files
- `config` directory defines a list of configuration for the program. Refer to `sample.yml` for more details.