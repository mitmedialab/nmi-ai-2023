# Beliefs about AI influence human-AI interaction and can be manipulated to increase perceived trustworthiness, empathy, and effectiveness
A repository for the paper "Beliefs about AI influence human-AI interaction and can be manipulated to increase perceived trustworthiness, empathy, and effectiveness," Nature Machine Intelligence 2023. 

## Notebooks 

Notebooks that have an Eliza and GPT version are essentially identical, and are merely copied for organization. 

### `0_data_pre_processing`

Processes the data from the CSV survey results and conversation transcripts and saves them into a new CSV. 

### `1_data_table` 
Extract statistics about the Likert questions on the survey, including which statistical tests were used, p-value, mean, standard deviation for each item split by assigned motives and by perceived motives.

### `2_demographics` 
A few simple vizualizations of demographics. 

### `3_convo_processing`
Processes conversation data from the pre-processed data and saves them into a new CSV. 

### `4_convo_viz`
Vizualizes conversation data, generating sentiment trend line plots and box plots, in addition to calculating regression statistics. Note that some of the labels do not have a neat appearance; the labels of the final figures in the paper often were often remade in another application. 

### `5_statistics_viz` 
Calculate statistics and generate bar charts for Likert items. The code for calculating the statistics is in `stat_process.py` in the Include folder. 

### `statistics_draft` 
The notebook used for drafting the code for `stat_process.py`. Includes some potentially useful information for clarity.

## Code 

`Include/stat_process.py` is used to calculate the statistics and generate plots, particularly for the Likert items of the survey. 

`requirements.txt` contains the packages necessary to download. 

`chatlog.js` a Google Apps Script to be run on Google Sheets to record data from a web interface. It can be added to Google Sheets from Extensions > Apps Script. 

### `Melu`

This folder includes HTML, Javascript, and CSS (SCSS) code. This was run on CodePen to create a web interface, which you can see here (API codes redacted): https://codepen.io/rliu3426/pen/bGQoPyd 

To use the CodePen, you will need to replace the `YOUR OPENAI API KEY HERE` with your OpenAI key. To collect the conversation data in a sheet, you will need to replace `YOUR SHEET API KEY HERE` with the URL generated from the Google Apps script.

## Data

Data is in the Results folder, split into Raw and Processed folders. 

### `Raw` 
CSV data for surveys and conversations. The Survey data is from Qualtrics, and the Chatlog data is the transcript data from Google Sheets.

### `Processed` 
Data processed and saved from the `0_data_pre_processing` notebooks, as well as conversation data processed and saved from the `3_convo_processing` notebooks. 

Note that conversation length labeled as "conversation turns" in the data and code, but this refers to the length. 
