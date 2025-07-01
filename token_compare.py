import os
import uuid
import requests
import json
import tiktoken
import google.generativeai as genai
from google.auth.transport.requests import Request
from google.auth import default
import re

# config.json에서 API 키 불러오기
# CLOVA: 클로바 API 키
# Claude: Anthropic API 키
# Gemini: Google Cloud API 키
with open("config.json", "r") as config_file:
    config = json.load(config_file)

# HyperCLOVA API를 사용하여 프롬프트의 토큰 수를 계산
def count_hyperclova_tokens(prompt: str):
    CLOVA_API_KEY = config["CLOVA_API_KEY"]
    MODEL_NAME = "HCX-003"
    if not CLOVA_API_KEY:
        print("[ERROR] CLOVA API key missing")
        return -1

    url = f"https://clovastudio.stream.ntruss.com/v1/api-tools/chat-tokenize/{MODEL_NAME}"
    headers = {
        "Authorization": f"Bearer {CLOVA_API_KEY}",
        "Content-Type": "application/json",
        "X-NCP-CLOVASTUDIO-REQUEST-ID": str(uuid.uuid4())
    }

    data = {
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        result = response.json()
        count = result["result"]["messages"][0]["count"]
        print("[DEBUG] HyperCLOVA tokens:", count)
        return count
    except Exception as e:
        print("[ERROR] HyperCLOVA token error:", str(e))
        return -1

# Tiktoken 패키지를 이용하여 OpenAI GPT-4o 모델의 토큰 수를 계산
def count_openai_gpt4o_tokens(prompt: str):
    try:
        encoding = tiktoken.encoding_for_model("gpt-4o")
        tokens = encoding.encode(prompt)
        print("[DEBUG] GPT-4o tokens:", len(tokens))
        return len(tokens)
    except Exception as e:
        print("[ERROR] GPT-4o token error:", str(e))
        return -1
    
# Tiktoken 패키지를 이용하여 OpenAI GPT-4 모델의 토큰 수를 계산
def count_openai_gpt4_tokens(prompt: str):
    try:
        encoding = tiktoken.encoding_for_model("gpt-4")
        tokens = encoding.encode(prompt)
        print("[DEBUG] GPT-4 tokens:", len(tokens))
        return len(tokens)
    except Exception as e:
        print("[ERROR] GPT-4 token error:", str(e))
        return -1

# 공식 토큰 계산 API를 이용하여 Anthropic Claude 모델의 토큰 수를 계산
def count_claude_tokens(prompt: str):
    import anthropic
    
    print(f"[DEBUG] Attempting to count tokens for prompt: '{prompt[:50]}...'")
    
    # Check if API key exists
    if "CLAUDE_API_KEY" not in config:
        print("[ERROR] CLAUDE_API_KEY not found in config")
        return -1
    
    api_key = config["CLAUDE_API_KEY"]
    if not api_key or api_key.strip() == "":
        print("[ERROR] CLAUDE_API_KEY is empty")
        return -1
    
    print(f"[DEBUG] API key found: {api_key[:10]}...")
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        print("[DEBUG] Anthropic client created successfully")
        
        # Try the beta token counting API first
        try:
            print("[DEBUG] Attempting beta token counting API...")
            response = client.beta.messages.count_tokens(
                betas=["token-counting-2024-11-01"],
                model="claude-3-5-sonnet-20241022",  # Using a more standard model
                messages=[{"role": "user", "content": prompt}]
            )
            print(f"[DEBUG] Beta API successful. Response: {response}")
            if hasattr(response, 'input_tokens'):
                tokens = response.input_tokens
                print(f"[DEBUG] Claude tokens (beta): {tokens}")
                return tokens
            else:
                print(f"[ERROR] Unexpected response format: {response}")
                return -1
                
        except Exception as beta_error:
            print(f"[DEBUG] Beta API failed: {beta_error}")
            print("[DEBUG] Trying regular token counting API...")
            
            # Fallback to regular API
            try:
                response = client.messages.count_tokens(
                    model="claude-3-5-sonnet-20241022",
                    messages=[{"role": "user", "content": prompt}]
                )
                print(f"[DEBUG] Regular API successful. Response: {response}")
                if hasattr(response, 'input_tokens'):
                    tokens = response.input_tokens
                    print(f"[DEBUG] Claude tokens (regular): {tokens}")
                    return tokens
                else:
                    print(f"[ERROR] Unexpected response format: {response}")
                    return -1
                    
            except Exception as regular_error:
                print(f"[ERROR] Regular API also failed: {regular_error}")
                raise regular_error
                
    except Exception as e:
        print(f"[ERROR] Claude token error details:")
        print(f"  Error type: {type(e).__name__}")
        print(f"  Error message: {str(e)}")
        
        # Check for specific error types
        if "authentication" in str(e).lower() or "api_key" in str(e).lower():
            print("  -> This appears to be an API key authentication issue")
        elif "rate_limit" in str(e).lower() or "quota" in str(e).lower():
            print("  -> This appears to be a rate limiting issue")
        elif "model" in str(e).lower():
            print("  -> This appears to be a model availability issue")
        elif "network" in str(e).lower() or "connection" in str(e).lower():
            print("  -> This appears to be a network connectivity issue")
        
        return -1

# 공식 토큰 계산 API를 이용하여 Google Gemini 모델의 토큰 수를 계산
def count_gemini_tokens(prompt: str):
    try:
        genai.configure(api_key=config["GEMINI_API_KEY"])
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        result = model.count_tokens([{"role": "user", "parts": [prompt]}])
        print("[DEBUG] Gemini tokens:", result.total_tokens)
        return result.total_tokens
    except Exception as e:
        print("[ERROR] Gemini token error:", str(e))
        return -1
    
# HuggingFace를 이용하여 LLaMA 모델의 토큰 수를 계산
def count_llama_tokens(prompt: str):
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B-Instruct")
        count = len(tokenizer.encode(prompt))
        print("[DEBUG] LLaMA tokens:", count)
        return count
    except Exception as e:
        print("[ERROR] LLaMA token error:", str(e))
        return -1

def count_korean_characters(prompt: str):
    korean_chars = re.findall(r'[\uac00-\ud7af]', prompt)  # Matches any Korean character in the Hangul syllabary
    return len(korean_chars)

# 여러 모델의 토큰 수를 반환하는 함수
def get_token_counts(prompt: str):
    counts = {
        "hyperclova": count_hyperclova_tokens(prompt),
        "gpt-4o": count_openai_gpt4o_tokens(prompt),
        "gpt-4": count_openai_gpt4_tokens(prompt),
        "claude": count_claude_tokens(prompt),
        "gemini": count_gemini_tokens(prompt),
        "llama": count_llama_tokens(prompt),
        "korean_chars": count_korean_characters(prompt)  # Add this line to count Korean characters
    }
    base = counts["hyperclova"]
    if base > 0:
        percentages = {k: round((v / base) * 100, 2) if v >= 0 else -1 for k, v in counts.items()}
    else:
        percentages = {k: -1 for k in counts}
    counts["percentages"] = percentages
    print("[DEBUG] Token counts with percentages:", counts)
    return counts
