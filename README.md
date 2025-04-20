# PromptClassifier: LLM Prompt Classifier & Comparator

![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)

**PromptClassifier** is a Python application that classifies user prompts into predefined categories (e.g., Politics, Mathematics) using zero-shot classification, enhances the prompts with contextual instructions based on the classification, and retrieves responses from the Groq API’s `llama3-70b-8192` Large Language Model (LLM). The application features a Tkinter-based graphical user interface (GUI) for easy interaction, allowing users to input prompts, view classifications, and compare LLM responses.

The project implements **Model-based Cross-Prompting (MCP)** to combine responses, though with a single model, the MCP response mirrors the individual response. This tool is ideal for researchers, developers, and enthusiasts interested in prompt engineering, LLM response analysis, and classification-based prompt enhancement.

## Features
- **Prompt Classification**: Uses the Hugging Face `facebook/bart-large-mnli` model for zero-shot classification to categorize prompts into primary (e.g., Politics) and secondary (e.g., Political Philosophy) classes.
- **Prompt Enhancement**: Adds contextual instructions based on classification to improve LLM responses.
- **LLM Response Retrieval**: Queries the Groq API’s `llama3-70b-8192` model for both vanilla (original prompt) and filtered (context-enhanced) responses.
- **Tkinter GUI**: Provides a user-friendly interface to input prompts and view classification and LLM responses.
- **MCP Integration**: Implements Model-based Cross-Prompting to combine responses (scalable for multiple models).

## Screenshot
![PromptClassifier GUI](screenshot.png)
*Enter a prompt, click "Run Comparison," and view the classification, vanilla, filtered, and MCP responses.*

## Requirements
To run this project, you’ll need the following:

### Software
- **Python 3.7+**: The code is written in Python and requires a version of 3.7 or higher.
- **Operating System**: Compatible with Windows, macOS, and Linux (Tkinter requires a graphical environment).

### Dependencies
Install the required Python packages using `pip`:
```bash
pip install transformers pandas requests

transformers: For zero-shot classification using Hugging Face’s facebook/bart-large-mnli model.

pandas: For data handling (optional, can be removed if not saving results to CSV).

requests: For making API calls to the Groq API.

tkinter: Comes with Python standard library for the GUI.

API Key
Groq API Key: Required to access the llama3-70b-8192 model. Sign up at Groq Console and create an API key.

Installation and Setup
Follow these steps to set up and run the project:
Clone the Repository:
bash

git clone https://github.com/your-username/PromptClassifier.git
cd PromptClassifier

Install Dependencies:
Install the required Python packages:
bash

pip install transformers pandas requests

Set Up Groq API Key:
Set your Groq API key as an environment variable:
Mac/Linux:
bash

export GROQ_API_KEY=your_groq_api_key

Windows:
cmd

setx GROQ_API_KEY your_groq_api_key

Restart your shell after setting the environment variable.

Alternatively, you can modify llm_configs in the code to pass the API key directly (not recommended for public repositories):
python

self.llm_configs = [
    {"provider": "groq", "model": "llama3-70b-8192", "api_key": "your_groq_api_key"}
]

Run the Application:
Execute the main script to launch the Tkinter GUI:
bash

python prompt_classifier.py

Usage
Launch the Application:
Run the script to open the Tkinter GUI:
bash

python prompt_classifier.py

Enter a Prompt:
In the GUI, type a prompt in the "Enter Prompt" field. Example:

Why the world is moving towards a right wing idealogy

Run Comparison:
Click the "Run Comparison" button to process the prompt. The application will:
Classify the prompt into primary and secondary categories (e.g., [Politics] Political Philosophy).

Query the Groq API (llama3-70b-8192) for a vanilla response (original prompt) and a filtered response (prompt with contextual instructions).

Display the MCP response (same as the individual response since only one model is used).

View Results:
The output area will show:
Prompt: The input prompt.

Subclasses: Classified categories with confidence scores.

Vanilla Responses: Direct response from the Groq API.

Filtered Responses: Response to the context-enhanced prompt.

MCP Responses: Combined response (mirrors the single model’s response).

Example Output
For the prompt "Why the world is moving towards a right wing idealogy":

Prompt:
Why the world is moving towards a right wing idealogy

Subclasses:
[Politics] Political Philosophy (0.82)

Vanilla Responses:
groq_llama3-70b-8192:
Economic uncertainty, immigration concerns, and disillusionment with progressive policies are driving a global shift towards right-wing ideologies...

Filtered Responses:
groq_llama3-70b-8192:
Focusing on Political Philosophy: The rise of right-wing ideologies can be traced to a resurgence of nationalist and conservative thought, often as a reaction to globalization...

MCP Vanilla Response:
Model 1 Response:
Economic uncertainty, immigration concerns, and disillusionment with progressive policies are driving a global shift towards right-wing ideologies...

MCP Filtered Response:
Model 1 Response:
Focusing on Political Philosophy: The rise of right-wing ideologies can be traced to a resurgence of nationalist and conservative thought, often as a reaction to globalization...

Code Explanation
Structure
LLMClient Class:
Handles API calls to the Groq API using the llama3-70b-8192 model.

Uses the requests library to send HTTP requests to https://api.groq.com/openai/v1/chat/completions.

Requires a valid Groq API key set as an environment variable (GROQ_API_KEY).

mcp_combine_responses Function:
Implements Model-based Cross-Prompting (MCP) to combine responses.

Currently, with one model, it mirrors the individual response but is scalable for multiple models.

Classification Setup:
Uses Hugging Face’s facebook/bart-large-mnli model for zero-shot classification.

Defines primary classes (e.g., Politics, Mathematics) and secondary classes (e.g., Political Philosophy, Algebra).

The classify_prompt function classifies prompts with adjusted thresholds (0.3) to ensure proper categorization.

The contextual_prompt function adds context based on classification (e.g., "You are an expert answering this question with special focus on: Political Philosophy").

run_comparison_tests Function:
Takes a prompt and llm_configs, classifies the prompt, queries the Groq API for vanilla and filtered responses, and returns results.

Tkinter GUI (LLMApp Class):
Provides a GUI with a prompt entry field, "Run Comparison" button, and scrollable output area.

Displays the prompt, subclasses, and responses from the Groq API.

Key Features
Zero-Shot Classification: Leverages facebook/bart-large-mnli to classify prompts without training data.

Prompt Enhancement: Adds contextual instructions to improve LLM responses.

Scalability: Designed to support multiple LLMs (though currently uses only Groq).

Troubleshooting
API Key Error:
If you see  API key for groq not found, ensure GROQ_API_KEY is set correctly.

Verify your key at Groq Console.

Rate Limits:
Groq’s free tier may have unpublished rate limits. If you encounter a 429 Too Many Requests error, wait a few minutes and retry.

No Subclasses Identified:
If the prompt isn’t classified (e.g., Subclasses: None), lower the thresholds in classify_prompt:
python

def classify_prompt(prompt: str, primary_threshold: float = 0.2, secondary_threshold: float = 0.2):

Debug classification scores:
python

primary_result = classifier(prompt, primary_classes, multi_label=True)
print(primary_result["labels"], primary_result["scores"])

Tkinter Issues:
Ensure your system has a graphical environment (e.g., X11 on Linux, native display on macOS/Windows).

If the output area is too small, adjust width and height in self.output_text:
python

self.output_text = scrolledtext.ScrolledText(self.frame, width=100, height=30, wrap=tk.WORD)

Future Improvements
Multiple Models: Add support for more Groq models (e.g., llama3-8b-8192) by extending llm_configs:
python

self.llm_configs = [
    {"provider": "groq", "model": "llama3-70b-8192"},
    {"provider": "groq", "model": "llama3-8b-8192"}
]

Enhanced MCP: Implement voting or weighted averaging for MCP when using multiple models.

GUI Features: Add a dropdown to select models or adjust classification thresholds dynamically.

Contributing
Contributions are welcome! Please fork the repository, make your changes, and submit a pull request. For major changes, open an issue to discuss your ideas.
License
This project is licensed under the MIT License. See the LICENSE file for details.
Contact
For questions or support, open an issue on GitHub or contact the repository owner.
Built by Manish

