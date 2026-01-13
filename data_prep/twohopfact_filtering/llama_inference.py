"""
Llama 3.1 8B Instruct inference using vLLM's OpenAI-compatible server.

This uses async methods so you can fire off multiple requests concurrently,
and vLLM will automatically batch them for efficient processing.

USAGE:
------
1. Start the vLLM server (one time, keep it running):
   $ python start_vllm_server.py
   
   This will take a few minutes to load the model, but then it stays running.

2. Check if server is ready:
   $ python check_server.py

3. Use the async API in your code:
   
   import asyncio
   from llama_inference import AsyncLlamaInference
   
   async def main():
       async with AsyncLlamaInference() as inferencer:
           # Single request
           answer = await inferencer.generate_single("What is 2+2?")
           
           # Many concurrent requests (vLLM batches them automatically)
           questions = ["Question 1?", "Question 2?", ...]
           answers = await inferencer.generate_batch(questions)
   
   asyncio.run(main())

See example_async_usage.py for more examples.
"""
import asyncio
from typing import List, Dict, Any, Optional
import httpx


class AsyncLlamaInference:
    """Async wrapper for Llama 3.1 8B Instruct inference via vLLM server."""
    
    def __init__(self, base_url: str = "http://localhost:8000/v1",
                 timeout: float = 300.0):
        """
        Initialize the async inference client.
        
        Args:
            base_url: Base URL of the vLLM server (OpenAI-compatible API)
            timeout: Request timeout in seconds (default 5 minutes for long generations)
        """
        self.base_url = base_url
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
        
        # Default generation parameters for Q&A
        self.default_params = {
            "temperature": 0.0,  # Greedy decoding for consistency
            "max_tokens": 256,   # Should be enough for most answers
            "top_p": 1.0,
        }
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    def format_messages(self, question: str) -> List[Dict[str, str]]:
        """
        Format a question using Llama 3.1 Instruct chat template.
        
        Args:
            question: The question to ask
            
        Returns:
            List of message dicts for the chat API
        """
        return [
            {"role": "system", "content": "You are a helpful assistant. Answer the question accurately and concisely."},
            {"role": "user", "content": question}
        ]
    
    async def generate_single(self, 
                             question: str, 
                             temperature: Optional[float] = None,
                             max_tokens: Optional[int] = None,
                             **kwargs) -> str:
        """
        Generate an answer for a single question asynchronously.
        
        Args:
            question: Question to answer
            temperature: Optional temperature override
            max_tokens: Optional max_tokens override
            **kwargs: Additional parameters for the API
            
        Returns:
            Generated answer (string)
        """
        params = self.default_params.copy()
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        params.update(kwargs)
        
        messages = self.format_messages(question)
        
        payload = {
            "model": "meta-llama/Llama-3.1-8B-Instruct",  # This should match the server
            "messages": messages,
            **params
        }
        
        response = await self.client.post(
            f"{self.base_url}/chat/completions",
            json=payload
        )
        response.raise_for_status()
        
        result = response.json()
        answer = result["choices"][0]["message"]["content"].strip()
        
        return answer
    
    async def generate_batch(self, 
                            questions: List[str],
                            temperature: Optional[float] = None,
                            max_tokens: Optional[int] = None,
                            **kwargs) -> List[str]:
        """
        Generate answers for multiple questions concurrently.
        
        This fires off all requests asynchronously, and vLLM will automatically
        batch them for efficient processing.
        
        Args:
            questions: List of questions to answer
            temperature: Optional temperature override
            max_tokens: Optional max_tokens override
            **kwargs: Additional parameters for the API
            
        Returns:
            List of generated answers (strings)
        """
        # Fire off all requests concurrently
        tasks = [
            self.generate_single(q, temperature, max_tokens, **kwargs) 
            for q in questions
        ]
        
        # Wait for all to complete
        answers = await asyncio.gather(*tasks)
        
        return answers


async def test_inference():
    """Test the async inference setup with a few examples."""
    print("\n" + "="*80)
    print("Testing Async Llama 3.1 8B Instruct Inference")
    print("="*80 + "\n")
    
    # Test questions
    test_questions = [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What is 2 + 2?",
    ]
    
    async with AsyncLlamaInference() as inferencer:
        print("Testing single inference...")
        answer = await inferencer.generate_single(test_questions[0])
        print(f"Q: {test_questions[0]}")
        print(f"A: {answer}\n")
        
        print("\n" + "-"*80)
        print("Running batch inference (all requests fired concurrently)...")
        print("-"*80 + "\n")
        
        answers = await inferencer.generate_batch(test_questions)
        
        for question, answer in zip(test_questions, answers):
            print(f"Q: {question}")
            print(f"A: {answer}")
            print()
    
    print("="*80)
    print("Async inference setup complete and working!")
    print("="*80 + "\n")


if __name__ == "__main__":
    print("\n" + "!"*80)
    print("IMPORTANT: Make sure the vLLM server is running first!")
    print("Start it with: python start_vllm_server.py")
    print("!"*80 + "\n")
    
    asyncio.run(test_inference())

