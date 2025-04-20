import os
import math
import pandas as pd
import requests
from transformers import pipeline
from typing import List, Dict, Tuple
import tkinter as tk
from tkinter import ttk, scrolledtext

# -------------------- 1. LLM Client --------------------

class LLMClient:
    def __init__(self, provider: str, api_key: str = None, model: str = None):
        self.provider = provider.lower()
        self.api_key = api_key or os.getenv(f"{self.provider.upper()}_API_KEY")
        self.model = model

        if not self.api_key:
            raise ValueError(f"❌ API key for {self.provider} not found in environment or passed directly.")

        if self.provider == "groq":
            self.model = self.model or "llama3-70b-8192"
            self.base_url = "https://api.groq.com/openai/v1"
        else:
            raise ValueError(f"❌ Unsupported provider: {self.provider}")

    def query(self, prompt: str, temperature: float = 0.7) -> str:
        try:
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            if self.provider == "groq":
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": 1024
                }
                response = requests.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"❌ Error: {e}"

# -------------------- 2. MCP Implementation --------------------

def mcp_combine_responses(responses: List[str]) -> str:
    if not responses:
        return "❌ No valid responses for MCP."
    valid_responses = [r for r in responses if not r.startswith("❌ Error")]
    if not valid_responses:
        return "❌ All responses contained errors."
    combined = "\n\n".join([f"Model {i+1} Response:\n{resp}" for i, resp in enumerate(valid_responses)])
    return combined

# -------------------- 3. Classification Setup --------------------

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)

primary_classes = [
    "Mathematics", "Philosophy", "Science", "Technology", "Literature", "History", "Politics", "Fashion", "Health",
    "Art", "Business", "Law", "Education", "Entertainment", "Sports", "Psychology", "Sociology", "Religion",
    "Finance", "Economics", "Environment", "Medicine", "Engineering", "AI", "Cybersecurity", "Ethics",
    "Geography", "Astronomy", "Linguistics", "Cultural Studies"
]

secondary_classes = {
    "Mathematics": ["Algebra", "Calculus", "Geometry", "Statistics", "Linear Algebra", "Number Theory"],
    "Philosophy": ["Ethics", "Metaphysics", "Epistemology", "Logic", "Aesthetics", "Political Philosophy"],
    "Science": ["Physics", "Chemistry", "Biology", "Astronomy", "Geology", "Environmental Science"],
    "Technology": ["AI", "Blockchain", "Cybersecurity", "Cloud Computing", "IoT", "Quantum Computing"],
    "Health": ["Mental Health", "Public Health", "Nutrition", "Medicine", "Epidemiology", "Genetics"],
    "Law": ["Criminal Law", "Civil Law", "Constitutional Law", "IP Law", "International Law"],
    "Business": ["Marketing", "Finance", "HR", "Operations", "Entrepreneurship", "Supply Chain"],
    "Education": ["Pedagogy", "Online Learning", "Curriculum Design", "EdTech", "Assessment"],
    "Psychology": ["Cognitive Psychology", "Behavioral Psychology", "Neuropsychology", "Developmental Psychology"],
    "Environment": ["Climate Change", "Sustainability", "Biodiversity", "Pollution", "Conservation"],
    "Finance": ["Stock Market", "Investment", "Risk Management", "Cryptocurrency", "Personal Finance"],
    "Engineering": ["Mechanical", "Electrical", "Civil", "Chemical", "Computer"],
    "Religion": ["Theology", "Comparative Religion", "Philosophy of Religion", "Mythology"],
    "Entertainment": ["Movies", "TV Shows", "Music", "Theatre", "Comics", "Pop Culture"],
}

def compute_entropy(probs: List[float]) -> float:
    return -sum(p * math.log(p + 1e-12) for p in probs)

def classify_prompt(prompt: str, primary_threshold: float = 0.3, secondary_threshold: float = 0.3) -> List[Tuple[str, str, float]]:
    primary_result = classifier(prompt, primary_classes, multi_label=True)
    entropy = compute_entropy(primary_result['scores'])
    refined_scores = [s * (1 - entropy) for s in primary_result['scores']]
    selected_primary = [label for label, score, _ in zip(primary_result["labels"], primary_result["scores"], refined_scores) if score > primary_threshold]

    selected_subclasses = []
    for domain in selected_primary:
        subcats = secondary_classes.get(domain, [])
        if subcats:
            sub_result = classifier(prompt, subcats, multi_label=True)
            for label, score in zip(sub_result["labels"], sub_result["scores"]):
                if score > secondary_threshold:
                    selected_subclasses.append((domain, label, score))

    return selected_subclasses

def contextual_prompt(prompt: str, subclasses: List[Tuple[str, str, float]]) -> str:
    subclass_labels = [label for _, label, _ in subclasses]
    context = f"You are an expert answering this question with special focus on: {', '.join(subclass_labels)}.\n\n"
    return context + prompt

# -------------------- 4. Comparison Logic --------------------

def run_comparison_tests(prompt: str, llm_configs: List[Dict[str, str]]) -> Dict:
    llm_clients = []

    for config in llm_configs:
        try:
            client = LLMClient(provider=config["provider"], api_key=config.get("api_key"), model=config.get("model"))
            llm_clients.append((client, config["provider"], config.get("model")))
        except Exception as e:
            print(f"❌ Failed to initialize {config['provider']}: {e}")

    if not llm_clients:
        raise ValueError("❌ No valid LLM clients initialized.")

    subclasses = classify_prompt(prompt)
    subclass_labels = "; ".join([f"[{d}] {s} ({round(sc, 2)})" for d, s, sc in subclasses]) or "None"
    filtered_prompt = contextual_prompt(prompt, subclasses)

    vanilla_responses = {}
    filtered_responses = {}
    for client, provider, model in llm_clients:
        model_key = f"{provider}_{model or 'default'}"
        vanilla_responses[model_key] = client.query(prompt)
        filtered_responses[model_key] = client.query(filtered_prompt)

    mcp_vanilla = mcp_combine_responses(list(vanilla_responses.values()))
    mcp_filtered = mcp_combine_responses(list(filtered_responses.values()))

    return {
        "prompt": prompt,
        "subclasses": subclass_labels,
        "vanilla_responses": vanilla_responses,
        "filtered_responses": filtered_responses,
        "mcp_vanilla": mcp_vanilla,
        "mcp_filtered": mcp_filtered
    }

# -------------------- 5. Tkinter Interface --------------------

class LLMApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LLM Prompt Classifier & Comparator")
        self.llm_configs = [
            {"provider": "groq", "model": "llama3-70b-8192"}
        ]

        # Create main frame
        self.frame = ttk.Frame(self.root, padding="10")
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Prompt entry
        ttk.Label(self.frame, text="Enter Prompt:").grid(row=0, column=0, sticky=tk.W)
        self.prompt_entry = ttk.Entry(self.frame, width=50)
        self.prompt_entry.grid(row=1, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))

        # Run button
        self.run_button = ttk.Button(self.frame, text="Run Comparison", command=self.run_comparison)
        self.run_button.grid(row=2, column=0, columnspan=2, pady=10)

        # Output area
        ttk.Label(self.frame, text="Output:").grid(row=3, column=0, sticky=tk.W)
        self.output_text = scrolledtext.ScrolledText(self.frame, width=80, height=20, wrap=tk.WORD)
        self.output_text.grid(row=4, column=0, columnspan=2, pady=5)

    def run_comparison(self):
        prompt = self.prompt_entry.get()
        if not prompt:
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, "Please enter a prompt.")
            return

        self.output_text.delete(1.0, tk.END)
        result = run_comparison_tests(prompt, self.llm_configs)

        # Display results
        output = f"Prompt:\n{result['prompt']}\n\n"
        output += f"Subclasses:\n{result['subclasses']}\n\n"
        
        output += "Vanilla Responses:\n"
        for model, response in result["vanilla_responses"].items():
            output += f"{model}:\n{response}\n\n"
        
        output += "Filtered Responses:\n"
        for model, response in result["filtered_responses"].items():
            output += f"{model}:\n{response}\n\n"
        
        output += f"MCP Vanilla Response:\n{result['mcp_vanilla']}\n\n"
        output += f"MCP Filtered Response:\n{result['mcp_filtered']}\n"

        self.output_text.insert(tk.END, output)

# -------------------- 6. Run Application --------------------

if __name__ == "__main__":
    root = tk.Tk()
    app = LLMApp(root)
    root.mainloop()