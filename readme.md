# Budget App Workflow

## 1. Configuration

- You'll need an OPENAI_API_KEY. Add this in config.py
- Load input data in data/input following the schema provided in the sample 
- add rules in auto_cat_rules.csv for rules-based (ie based on keywords) categorization
- list desired categories and subcategories in the categorizer.py file 

## 2. Run Script to Assign Categories

- run the app.py streamlit app to assign categories, subcategories, and exclude_flag for each row
- this also includes an eval option. for this, you will need to give manually-assigned records (ie true answers) in the data/eval folder for comparison

## 3. Add to Google Sheet for Analysis

- once you have the assigned output records, you can take it from there to analyze! 
- see here for a template: https://docs.google.com/spreadsheets/d/1x1LyMDQfW8I5xIvf8ZtfOfjIUG8cTwpK3enO3IvlCcU/edit?usp=sharing
