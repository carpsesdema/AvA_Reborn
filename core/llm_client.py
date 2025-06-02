# core/llm_client.py - Enhanced with Streaming Support

import os
import requests
import json
import asyncio
from typing import AsyncGenerator, Optional


class LLMClient:
    """
    SINGLE RESPONSIBILITY: Handle LLM API calls with streaming support
    Does NOT handle workflow, UI, file operations, or other logic
    """

    def __init__(self):
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    def chat(self, prompt: str) -> str:
        """
        Send prompt to LLM and return complete response (synchronous)
        Tries Gemini first (user preference), then fallbacks
        """

        # Try Gemini first (user's preference)
        if self.gemini_key:
            response = self._call_gemini(prompt)
            if response:
                return response

        # Try OpenAI as backup
        if self.openai_key:
            response = self._call_openai(prompt)
            if response:
                return response

        # Try Anthropic as backup
        if self.anthropic_key:
            response = self._call_anthropic(prompt)
            if response:
                return response

        # Try Ollama (local) as final backup
        response = self._call_ollama(prompt)
        if response:
            return response

        # No LLM available
        return self._fallback_response(prompt)

    async def stream_chat(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Stream response from LLM token by token (asynchronous generator)
        Tries same fallback order as chat()
        """

        # Try Gemini streaming first
        if self.gemini_key:
            async for chunk in self._stream_gemini(prompt):
                if chunk:
                    yield chunk
            return

        # Try OpenAI streaming
        if self.openai_key:
            async for chunk in self._stream_openai(prompt):
                if chunk:
                    yield chunk
            return

        # Try Anthropic streaming
        if self.anthropic_key:
            async for chunk in self._stream_anthropic(prompt):
                if chunk:
                    yield chunk
            return

        # Try Ollama streaming
        async for chunk in self._stream_ollama(prompt):
            if chunk:
                yield chunk
            return

        # Fallback: yield complete response as single chunk
        fallback_response = self._fallback_response(prompt)
        yield fallback_response

    def _call_gemini(self, prompt: str) -> str:
        """Call Gemini API (synchronous)"""
        try:
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_key}",
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.1,
                        "maxOutputTokens": 4000
                    }
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                if ("candidates" in result and
                        len(result["candidates"]) > 0 and
                        "content" in result["candidates"][0] and
                        "parts" in result["candidates"][0]["content"] and
                        len(result["candidates"][0]["content"]["parts"]) > 0):
                    return result["candidates"][0]["content"]["parts"][0]["text"]

            print(f"Gemini API error: {response.status_code}")
            return None

        except Exception as e:
            print(f"Gemini failed: {e}")
            return None

    async def _stream_gemini(self, prompt: str) -> AsyncGenerator[str, None]:
        """Stream from Gemini API"""
        try:
            # Gemini doesn't have true streaming in their REST API yet
            # So we simulate streaming by chunking the response
            response = await asyncio.get_event_loop().run_in_executor(
                None, self._call_gemini, prompt
            )

            if response:
                # Simulate streaming by yielding chunks
                words = response.split()
                for i in range(0, len(words), 3):  # 3 words per chunk
                    chunk = " ".join(words[i:i + 3]) + " "
                    yield chunk
                    await asyncio.sleep(0.05)  # Small delay for streaming effect

        except Exception as e:
            print(f"Gemini streaming failed: {e}")
            return

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API (synchronous)"""
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openai_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1
                },
                timeout=30
            )

            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]

            print(f"OpenAI API error: {response.status_code}")
            return None

        except Exception as e:
            print(f"OpenAI failed: {e}")
            return None

    async def _stream_openai(self, prompt: str) -> AsyncGenerator[str, None]:
        """Stream from OpenAI API"""
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.openai_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": "gpt-4o-mini",
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0.1,
                            "stream": True
                        }
                ) as response:
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                break
                            try:
                                chunk_data = json.loads(data)
                                if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                                    delta = chunk_data["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        yield delta["content"]
                            except json.JSONDecodeError:
                                continue

        except Exception as e:
            print(f"OpenAI streaming failed: {e}")
            # Fallback to non-streaming
            response = await asyncio.get_event_loop().run_in_executor(
                None, self._call_openai, prompt
            )
            if response:
                yield response

    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API (synchronous)"""
        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.anthropic_key,
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01"
                },
                json={
                    "model": "claude-3-haiku-20240307",
                    "max_tokens": 4000,
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=30
            )

            if response.status_code == 200:
                return response.json()["content"][0]["text"]

            print(f"Anthropic API error: {response.status_code}")
            return None

        except Exception as e:
            print(f"Anthropic failed: {e}")
            return None

    async def _stream_anthropic(self, prompt: str) -> AsyncGenerator[str, None]:
        """Stream from Anthropic API"""
        try:
            # Anthropic streaming implementation would go here
            # For now, fallback to chunked response
            response = await asyncio.get_event_loop().run_in_executor(
                None, self._call_anthropic, prompt
            )

            if response:
                # Simulate streaming by chunking
                words = response.split()
                for i in range(0, len(words), 4):  # 4 words per chunk
                    chunk = " ".join(words[i:i + 4]) + " "
                    yield chunk
                    await asyncio.sleep(0.03)

        except Exception as e:
            print(f"Anthropic streaming failed: {e}")
            return

    def _call_ollama(self, prompt: str) -> str:
        """Call local Ollama API (synchronous)"""
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "qwen2.5-coder",
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60
            )

            if response.status_code == 200:
                return response.json()["response"]

            return None

        except Exception as e:
            print(f"Ollama failed: {e}")
            return None

    async def _stream_ollama(self, prompt: str) -> AsyncGenerator[str, None]:
        """Stream from Ollama API"""
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.post(
                        "http://localhost:11434/api/generate",
                        json={
                            "model": "qwen2.5-coder",
                            "prompt": prompt,
                            "stream": True
                        }
                ) as response:
                    async for line in response.content:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            if "response" in data:
                                yield data["response"]
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            print(f"Ollama streaming failed: {e}")
            # Fallback to non-streaming
            response = await asyncio.get_event_loop().run_in_executor(
                None, self._call_ollama, prompt
            )
            if response:
                yield response

    def _fallback_response(self, prompt: str) -> str:
        """Fallback response when no LLM is available"""
        return f"""
# Error: No LLM service available

# To fix this, set up at least one API key:
# export GEMINI_API_KEY=your-key-here
# export OPENAI_API_KEY=sk-your-key-here  
# export ANTHROPIC_API_KEY=sk-ant-your-key-here

# Or install Ollama locally:
# curl -fsSL https://ollama.ai/install.sh | sh
# ollama pull qwen2.5-coder

# Sample code for your request: {prompt}

def example_function():
    '''
    Generated placeholder for: {prompt}
    Replace this with actual implementation once LLM is working.
    '''
    print("Hello from AvA!")
    pass

if __name__ == "__main__":
    example_function()
"""

    def get_available_models(self) -> list:
        """Get list of available LLM services"""
        available = []

        if self.gemini_key:
            available.append("Gemini (Primary)")
        if self.openai_key:
            available.append("OpenAI (Backup)")
        if self.anthropic_key:
            available.append("Anthropic (Backup)")

        # Check Ollama
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                available.append("Ollama (Local)")
        except:
            pass

        return available if available else ["No LLM services available"]