"""
AnuRAG: LLM Provider Abstraction Layer
Allows seamless switching between Gemini and Claude (Anthropic) for chat/reasoning.

The vector database, embeddings, and search are INDEPENDENT of this module.
This module handles ONLY the chat/reasoning LLM calls.

Usage:
    from llm_provider import get_llm_provider
    provider = get_llm_provider()
    response_text = provider.generate(prompt, system_instruction="You are an expert...")
"""

import os
import io
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# ABSTRACT BASE CLASS
# =============================================================================

class LLMProvider(ABC):
    """Abstract base class for LLM providers (Gemini, Claude, etc.)."""

    @abstractmethod
    def generate(
        self,
        contents: Any,
        system_instruction: str = "",
        temperature: float = 0.0,
        max_output_tokens: int = 4096,
        images: Optional[List[Any]] = None,
    ) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            contents: The conversation contents. Can be:
                - A plain string (single turn)
                - A list of {"role": "user"/"assistant", "content": str} dicts (multi-turn)
            system_instruction: System prompt / instruction
            temperature: Sampling temperature
            max_output_tokens: Max tokens in response
            images: Optional list of PIL Image objects for multimodal input
            
        Returns:
            The model's response text.
        """
        pass

    @abstractmethod
    def generate_with_history(
        self,
        history: List[Any],
        new_message: str,
        system_instruction: str = "",
        temperature: float = 0.0,
        max_output_tokens: int = 4096,
        images: Optional[List[Any]] = None,
    ) -> Tuple[str, List[Any]]:
        """
        Generate a response given conversation history, and return updated history.
        
        Args:
            history: Provider-specific conversation history (opaque to caller)
            new_message: The new user message
            system_instruction: System prompt
            temperature: Sampling temperature  
            max_output_tokens: Max output tokens
            images: Optional PIL Images for this turn
            
        Returns:
            (response_text, updated_history)
        """
        pass

    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g., 'gemini', 'claude')."""
        pass

    @abstractmethod
    def model_name(self) -> str:
        """Return the specific model being used."""
        pass


# =============================================================================
# GEMINI PROVIDER
# =============================================================================

class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider using google-genai package."""

    def __init__(self, model: str = None):
        from config import GEMINI_CHAT_MODEL
        self.model = model or GEMINI_CHAT_MODEL
        self._client = None
        self._init_client()

    def _init_client(self):
        try:
            from google import genai
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not set")
            self._client = genai.Client(api_key=api_key)
            self._use_new_api = True
        except ImportError:
            try:
                import google.generativeai as genai_old
                api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
                if not api_key:
                    raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not set")
                genai_old.configure(api_key=api_key)
                self._genai_old = genai_old
                self._use_new_api = False
            except ImportError:
                raise ImportError("No Gemini package installed. Run: pip install google-genai")

    def provider_name(self) -> str:
        return "gemini"

    def model_name(self) -> str:
        return self.model

    def generate(
        self,
        contents: Any,
        system_instruction: str = "",
        temperature: float = 0.0,
        max_output_tokens: int = 4096,
        images: Optional[List[Any]] = None,
    ) -> str:
        if self._use_new_api:
            return self._generate_new_api(contents, system_instruction, temperature, max_output_tokens, images)
        else:
            return self._generate_old_api(contents, system_instruction, temperature, max_output_tokens, images)

    def _generate_new_api(self, contents, system_instruction, temperature, max_output_tokens, images):
        from google.genai import types

        # Build parts
        parts = []
        if images:
            for img in images:
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='PNG')
                img_bytes.seek(0)
                parts.append(types.Part.from_bytes(data=img_bytes.read(), mime_type="image/png"))

        if isinstance(contents, str):
            parts.append(types.Part(text=contents))
            api_contents = [types.Content(role="user", parts=parts)]
        elif isinstance(contents, list):
            # Multi-turn: list of Content objects or dicts
            api_contents = contents
            # Append images to the last user message if provided
            if images and parts:
                # Images already in parts list, but text wasn't added yet
                pass
        else:
            parts.append(types.Part(text=str(contents)))
            api_contents = [types.Content(role="user", parts=parts)]

        gen_config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            system_instruction=system_instruction if system_instruction else None,
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode='NONE')
            ),
        )

        # Add thinking config for models that support it
        if 'pro' in self.model.lower() or '3' in self.model:
            try:
                gen_config.thinking_config = types.ThinkingConfig(thinking_level='low')
            except Exception:
                pass

        response = self._client.models.generate_content(
            model=self.model,
            contents=api_contents,
            config=gen_config
        )

        try:
            return response.text or ""
        except Exception:
            try:
                if response.candidates and response.candidates[0].content.parts:
                    return response.candidates[0].content.parts[0].text or ""
            except Exception:
                pass
            return "Error: Could not extract response text from Gemini"

    def _generate_old_api(self, contents, system_instruction, temperature, max_output_tokens, images):
        model = self._genai_old.GenerativeModel(
            self.model,
            system_instruction=system_instruction if system_instruction else None
        )
        prompt = contents if isinstance(contents, str) else str(contents)

        content_parts = []
        if images:
            content_parts.extend(images)
        content_parts.append(prompt)

        response = model.generate_content(
            content_parts,
            generation_config=self._genai_old.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens
            )
        )
        try:
            return response.text or ""
        except Exception:
            return "Error: Could not extract response text from Gemini"

    def generate_with_history(
        self,
        history: List[Any],
        new_message: str,
        system_instruction: str = "",
        temperature: float = 0.0,
        max_output_tokens: int = 4096,
        images: Optional[List[Any]] = None,
    ) -> Tuple[str, List[Any]]:
        if self._use_new_api:
            return self._generate_with_history_new(history, new_message, system_instruction, temperature, max_output_tokens, images)
        else:
            return self._generate_with_history_old(history, new_message, system_instruction, temperature, max_output_tokens, images)

    def _generate_with_history_new(self, history, new_message, system_instruction, temperature, max_output_tokens, images):
        from google.genai import types

        contents = list(history)  # copy existing history

        # Build current user message
        parts = []
        if images:
            for img in images:
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='PNG')
                img_bytes.seek(0)
                parts.append(types.Part.from_bytes(data=img_bytes.read(), mime_type="image/png"))
        parts.append(types.Part(text=new_message))
        contents.append(types.Content(role="user", parts=parts))

        gen_config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            system_instruction=system_instruction if system_instruction else None,
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode='NONE')
            ),
        )
        if 'pro' in self.model.lower() or '3' in self.model:
            try:
                gen_config.thinking_config = types.ThinkingConfig(thinking_level='low')
            except Exception:
                pass

        response = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=gen_config
        )

        try:
            response_text = response.text or ""
        except Exception:
            try:
                if response.candidates and response.candidates[0].content.parts:
                    response_text = response.candidates[0].content.parts[0].text or ""
                else:
                    response_text = "Error: Could not extract response text"
            except Exception:
                response_text = "Error: Could not extract response text"

        # Update history
        user_content = types.Content(role="user", parts=[types.Part(text=new_message)])
        model_content = types.Content(role="model", parts=[types.Part(text=response_text)])
        updated_history = list(history)
        updated_history.append(user_content)
        updated_history.append(model_content)

        return response_text, updated_history

    def _generate_with_history_old(self, history, new_message, system_instruction, temperature, max_output_tokens, images):
        # Old API: rebuild full prompt from history
        model = self._genai_old.GenerativeModel(
            self.model,
            system_instruction=system_instruction if system_instruction else None
        )
        chat = model.start_chat(history=history or [])

        content_parts = []
        if images:
            content_parts.extend(images)
        content_parts.append(new_message)

        response = chat.send_message(
            content_parts if len(content_parts) > 1 else new_message,
            generation_config=self._genai_old.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens
            )
        )
        try:
            response_text = response.text or ""
        except Exception:
            response_text = "Error: Could not extract response text"

        return response_text, chat.history


# =============================================================================
# CLAUDE (ANTHROPIC) PROVIDER
# =============================================================================

class ClaudeProvider(LLMProvider):
    """
    Anthropic Claude LLM provider.
    
    Recommended models for analog IC design (highly intelligent reasoning):
    
    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
    â”‚ Model                   â”‚ Input/1M â”‚ Output/1Mâ”‚ Best For                â”‚
    â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
    â”‚ claude-opus-4-20250514  â”‚  $15.00  â”‚  $75.00  â”‚ Maximum reasoning power â”‚
    â”‚ claude-sonnet-4-20250514â”‚   $3.00  â”‚  $15.00  â”‚ Best cost/intelligence  â”‚
    â”‚ claude-3-5-haiku-latest â”‚   $0.80  â”‚   $4.00  â”‚ Fast, cheaper tasks     â”‚
    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
    
    For AnuRAG's analog IC design tasks (physics reasoning + code generation):
    â†’ claude-sonnet-4-20250514 is the recommended sweet spot.
    â†’ claude-opus-4-20250514 for maximum quality on complex topology analysis.
    """

    def __init__(self, model: str = None):
        from config import CLAUDE_CHAT_MODEL
        self.model = model or CLAUDE_CHAT_MODEL
        self._client = None
        self._init_client()

    def _init_client(self):
        try:
            import anthropic
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY not set. Get your key from: https://console.anthropic.com/settings/keys"
                )
            self._client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
            )

    def provider_name(self) -> str:
        return "claude"

    def model_name(self) -> str:
        return self.model

    def generate(
        self,
        contents: Any,
        system_instruction: str = "",
        temperature: float = 0.0,
        max_output_tokens: int = 4096,
        images: Optional[List[Any]] = None,
    ) -> str:
        import anthropic

        # Build messages list
        messages = []

        if isinstance(contents, str):
            # Single turn
            content_parts = self._build_content_parts(contents, images)
            messages.append({"role": "user", "content": content_parts})
        elif isinstance(contents, list):
            # Could be list of dicts with role/content, or provider-specific objects
            messages = self._normalize_messages(contents, images)
        else:
            content_parts = self._build_content_parts(str(contents), images)
            messages.append({"role": "user", "content": content_parts})

        # Claude API call
        kwargs = {
            "model": self.model,
            "max_tokens": max_output_tokens,
            "messages": messages,
        }
        if system_instruction:
            kwargs["system"] = system_instruction
        if temperature > 0:
            kwargs["temperature"] = temperature

        # Use extended thinking for complex reasoning if model supports it
        # Extended thinking requires temperature=1 and is incompatible with explicit temperature
        if self._supports_extended_thinking():
            # For complex analog IC design, enable extended thinking
            kwargs["temperature"] = 1  # Required for extended thinking
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": 5000  # Conservative thinking budget
            }
            # Extended thinking needs higher max_tokens to accommodate thinking + response
            kwargs["max_tokens"] = max(max_output_tokens, 16000)

        response = self._client.messages.create(**kwargs)

        # Extract text from response (skip thinking blocks)
        return self._extract_response_text(response)

    def generate_with_history(
        self,
        history: List[Any],
        new_message: str,
        system_instruction: str = "",
        temperature: float = 0.0,
        max_output_tokens: int = 4096,
        images: Optional[List[Any]] = None,
    ) -> Tuple[str, List[Any]]:
        import anthropic

        # History for Claude is a list of {"role": "user"/"assistant", "content": ...} dicts
        messages = list(history) if history else []

        # Add new user message
        content_parts = self._build_content_parts(new_message, images)
        messages.append({"role": "user", "content": content_parts})

        kwargs = {
            "model": self.model,
            "max_tokens": max_output_tokens,
            "messages": messages,
        }
        if system_instruction:
            kwargs["system"] = system_instruction
        if temperature > 0:
            kwargs["temperature"] = temperature

        if self._supports_extended_thinking():
            kwargs["temperature"] = 1
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": 5000
            }
            kwargs["max_tokens"] = max(max_output_tokens, 16000)

        response = self._client.messages.create(**kwargs)
        response_text = self._extract_response_text(response)

        # Update history (store only text, not thinking blocks)
        updated_history = list(history) if history else []
        # Simplify the user content for history (text only, no images on repeat)
        updated_history.append({"role": "user", "content": new_message})
        updated_history.append({"role": "assistant", "content": response_text})

        return response_text, updated_history

    def _supports_extended_thinking(self) -> bool:
        """Check if current model supports extended thinking."""
        # Extended thinking is supported on Claude 3.5 Sonnet and newer
        thinking_models = ['claude-sonnet-4', 'claude-opus-4', 'claude-3-7-sonnet']
        return any(m in self.model for m in thinking_models)

    def _build_content_parts(self, text: str, images: Optional[List[Any]] = None) -> list:
        """Build Claude content parts (text + optional images)."""
        import base64

        parts = []
        if images:
            for img in images:
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='PNG')
                img_bytes.seek(0)
                b64_data = base64.standard_b64encode(img_bytes.read()).decode('utf-8')
                parts.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": b64_data,
                    }
                })
        parts.append({"type": "text", "text": text})
        return parts

    def _normalize_messages(self, contents: list, images: Optional[List[Any]] = None) -> list:
        """Normalize various message formats to Claude's expected format."""
        messages = []
        for item in contents:
            if isinstance(item, dict):
                role = item.get("role", "user")
                content = item.get("content", "")
                # Normalize role names
                if role == "model":
                    role = "assistant"
                messages.append({"role": role, "content": content})
            elif hasattr(item, 'role') and hasattr(item, 'parts'):
                # Gemini Content objects â€” convert
                role = "assistant" if item.role == "model" else item.role
                text_parts = []
                for part in (item.parts or []):
                    if hasattr(part, 'text') and part.text:
                        text_parts.append(part.text)
                if text_parts:
                    messages.append({"role": role, "content": " ".join(text_parts)})
        
        # Add images to the last user message if provided
        if images and messages:
            for i in range(len(messages) - 1, -1, -1):
                if messages[i]["role"] == "user":
                    content_parts = self._build_content_parts(messages[i]["content"], images)
                    messages[i]["content"] = content_parts
                    break

        return messages

    def _extract_response_text(self, response) -> str:
        """Extract text from Claude response, skipping thinking blocks."""
        text_parts = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            # Skip "thinking" blocks â€” those are internal reasoning
        return "\n".join(text_parts) if text_parts else "Error: Empty response from Claude"


# =============================================================================
# PROVIDER FACTORY
# =============================================================================

_provider_instance: Optional[LLMProvider] = None


def get_llm_provider(force_provider: str = None, force_model: str = None) -> LLMProvider:
    """
    Get or create the LLM provider singleton.
    
    Args:
        force_provider: Override the configured provider ("gemini" or "claude")
        force_model: Override the configured model name
        
    Returns:
        LLMProvider instance
    """
    global _provider_instance

    from config import LLM_PROVIDER

    provider_name = (force_provider or LLM_PROVIDER).lower().strip()

    # Return cached instance if same provider
    if _provider_instance is not None and _provider_instance.provider_name() == provider_name:
        return _provider_instance

    if provider_name == "claude" or provider_name == "anthropic":
        _provider_instance = ClaudeProvider(model=force_model)
        print(f"ðŸ¤– LLM Provider: Claude ({_provider_instance.model_name()})")
    elif provider_name == "gemini" or provider_name == "google":
        _provider_instance = GeminiProvider(model=force_model)
        print(f"ðŸ¤– LLM Provider: Gemini ({_provider_instance.model_name()})")
    else:
        raise ValueError(
            f"Unknown LLM provider: '{provider_name}'. "
            f"Set LLM_PROVIDER to 'gemini' or 'claude' in your .env file."
        )

    return _provider_instance


def reset_provider():
    """Reset the cached provider instance (useful for testing or switching)."""
    global _provider_instance
    _provider_instance = None
