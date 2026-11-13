"""
utility function for prompting OpenAI models
"""

import os
import logging
import requests
from openai import OpenAI, AzureOpenAI
import time
from tenacity import retry, wait_random_exponential, stop_after_attempt

# Suppress verbose HTTP request logs from the OpenAI client's httpx backend
logging.getLogger("httpx").setLevel(logging.WARNING)

def construct_prompt_message(system_prompt, user_prompt, few_shot_prompt=None):
    """
    Construct a prompt message for OpenAI ChatCompletion model
    """
    prompt_message = []
    prompt_message.append({"role": "system", "content": system_prompt})
    if few_shot_prompt:
        for example_num, example in few_shot_prompt.items():
            prompt_message.append({"role": "user", "content": example["User"]})
            prompt_message.append(
                {"role": "assistant", "content": example["Assistant"]}
            )

    prompt_message.append({"role": "user", "content": user_prompt})

    return prompt_message

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def prompt_openai(prompt_messages, max_tokens=2000, temperature=1.0, top_p=0.95, model="gpt-4.1", debug=False):        
    client = OpenAI()
    if debug:
        print(f"DEBUG: Attempting to call OpenAI API with model: {model}")
    
    try:
        # Use max_tokens for GPT-4 models and max_completion_tokens for GPT-5/o1 models
        if "gpt-4" in model:
            if debug:
                print(f"DEBUG: Using max_tokens parameter for {model}")
            completion = client.chat.completions.create(
                model=model,
                messages=prompt_messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
        else:
            if debug:
                print(f"DEBUG: Using max_completion_tokens parameter for {model}")
            completion = client.chat.completions.create(
                model=model,
                messages=prompt_messages,
                max_completion_tokens=max_tokens,
            )
        
        if debug:
            print(f"DEBUG: API call successful for {model}")
        
        # Check finish reason
        finish_reason = completion.choices[0].finish_reason
        if debug:
            print(f"DEBUG: Finish reason: {finish_reason}")
        
        response_content = completion.choices[0].message.content
        if debug:
            print(f"DEBUG: Response content length: {len(response_content) if response_content else 0}")
            if not response_content:
                print(f"DEBUG: Response content is None or empty!")
                print(f"DEBUG: Full completion object: {completion}")
        
    except Exception as e:
        if debug:
            print(f"DEBUG: OpenAI API Error for model {model}:")
            print(f"DEBUG: Error type: {type(e).__name__}")
            print(f"DEBUG: Error message: {str(e)}")
            
            # Print additional details if it's an OpenAI API error
            if hasattr(e, 'response'):
                print(f"DEBUG: HTTP status code: {e.response.status_code if e.response else 'N/A'}")
            if hasattr(e, 'code'):
                print(f"DEBUG: Error code: {e.code}")
            if hasattr(e, 'param'):
                print(f"DEBUG: Error param: {e.param}")
            
        # Re-raise the exception so the retry mechanism can work
        raise
    
    return response_content

        

@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(10))
def prompt_llama_router(prompt_messages, max_tokens=2000, temperature=0.0, top_p=1.0, model='llama'):
    # Retrieve the OpenRouter API key
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set.")

    # Define the base URL and headers
    base_url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json",
        # Optional headers for OpenRouter rankings
        "HTTP-Referer": "https://your-site-url.com",  # Replace with your site URL
        "X-Title": "Evaluation",  # Replace with your app name
    }

    if 'llama70b:free' in model:
        model = "meta-llama/llama-3.3-70b-instruct:free"
        order = ["venice/fp8"] 
    elif 'llama70b' in model:
        model = "meta-llama/llama-3.3-70b-instruct"
        order = ["groq", "friendli", "google-vertex"] 
    elif 'qwen30ba3b' in model:
        model = "qwen/qwen3-30b-a3b:free"
        order = ["chutes"]
    elif 'qwen3-235ba22b' in model:
        model = "qwen/qwen3-235b-a22b"
        order = ["fireworks"]
    elif 'llama4scout' in model:
        model = "meta-llama/llama-4-scout"
        order = ["groq", "friendli"]
    elif 'llama4maverick' in model:
        model = "meta-llama/llama-4-maverick"
        order = ["groq"]
  

    # Define the request payload
    body = {
        "model": model,
        "messages": prompt_messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "provider": {
            "order": order,  # Provider prioritization
            "allow_fallbacks": False,  # Disable fallbacks
        },
    }

    # Make the POST request
    response = requests.post(base_url, headers=headers, json=body)

    # Handle errors
    if response.status_code != 200:
        raise RuntimeError(
            f"Request failed with status {response.status_code}: {response.text}"
        )

    # Return the generated response
    return response.json()["choices"][0]["message"]["content"]
