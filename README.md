# LLM Token Comparator

Compare token counts across major Large Language Model (LLM) providers — including OpenAI, Claude, Gemini, LLaMA, and Naver HyperCLOVA — to better understand prompt cost, length, and tokenization strategy.

Especially useful for analyzing Korean-language tokenization, where HyperCLOVA X serves as a leading benchmark.

---

## Features

- Compare token counts across multiple LLM providers
- Visualize how each model tokenizes the same input
- Estimate prompt costs (optional)
- Paste text or upload files (coming soon)
- Dark/light mode support
- Open-source and developer-ready

---

## Supported Models

| Provider     | Model(s)                  | Tokenizer Method             |
|--------------|---------------------------|-------------------------------|
| OpenAI       | GPT-3.5, GPT-4            | `tiktoken`                   |
| Anthropic    | Claude 1/2/3              | Approximate logic            |
| Google       | Gemini                    | `sentencepiece`-based        |
| Meta         | LLaMA 2 / 3               | HuggingFace tokenizer        |
| Naver CLOVA  | HyperCLOVA X (bge-m3, etc.) | Optional internal tokenizer |

---

## Getting Started

### 1. Clone the Repository

<pre>
git clone https://github.com/IronhawkReigns/TokenCalculator.git
cd llm-token-comparator
</pre>

### 2. Install Dependencies

<pre>
pip install -r requirements.txt
</pre>

### 3. Run the App

<pre>
python app.py
</pre>

Or, if you're using FastAPI:

<pre>
uvicorn app:app --reload
</pre>

---

## Docker Support (Optional)

To build and run with Docker:

<pre>
docker build -t token-comparator .
docker run -p 8000:8000 token-comparator
</pre>

---

## Project Structure

llm-token-comparator/
├── app.py                   # Main backend logic
├── tokenizer/               # Tokenizer wrappers
├── static/                  # Frontend assets
├── templates/               # UI templates (Flask/Jinja)
├── requirements.txt
├── Dockerfile
└── README.md

---

## Sample Output

| Model     | Token Count |
|-----------|-------------|
| GPT-4     | 244         |
| Claude 2  | 212         |
| Gemini    | 198         |
| LLaMA 3   | 232         |
| CLOVA X   | 205         |

---

## Disclaimer

This tool is for developer and research use only. Tokenizer logic may differ slightly from official APIs. Always verify against provider documentation for billing-critical cases.

---

## Contributing

Pull requests and tokenizer plugin contributions are welcome!  
Want to add a provider? Follow the `/tokenizer/` pattern and open a PR.

---

## Author

Built by [YJ Shin](https://github.com/IronhawkReigns)  
Computer Science student @ Georgia Tech and Naver Cloud intern.

> This tool was created to help developers better understand how LLMs tokenize input — essential for optimizing prompts, controlling cost, and debugging strange model behavior.
