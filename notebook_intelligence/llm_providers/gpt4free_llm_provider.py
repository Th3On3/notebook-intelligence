# Copyright (c) Mehmet Bektas <mbektasgh@outlook.com>

import json
from typing import Any
from notebook_intelligence.api import ChatModel, EmbeddingModel, InlineCompletionModel, LLMProvider, CancelToken, ChatResponse, CompletionContext, LLMProviderProperty
from g4f.client import Client as Gpt4Free
from g4f.Provider import ProviderType, ProviderUtils, __providers__


DEFAULT_CONTEXT_WINDOW = 4096

LIST_MODELS = []
LIST_INLINE_MODELS = []

class Gpt4FreeChatModel(ChatModel):
    def __init__(self, provider, model_id):
        super().__init__(provider)
        self._provider = provider
        self.model_id = model_id
        self._properties = [
            #LLMProviderProperty("api_key", "API key", "API key", "", True),
            LLMProviderProperty("context_window", "Context window", "Context window length", "", True),
        ]

    @property
    def id(self) -> str:
        return self.model_id
    
    @property
    def name(self) -> str:
        return self.model_id
    
    @property
    def context_window(self) -> int:
        try:
            context_window_prop = self.get_property("context_window")
            if context_window_prop is not None:
                context_window = int(context_window_prop.value)
            return context_window
        except:
            return DEFAULT_CONTEXT_WINDOW

    def completions(self, messages: list[dict], tools: list[dict] = None, response: ChatResponse = None, cancel_token: CancelToken = None, options: dict = {}) -> Any:
        stream = response is not None
        model_id = self.model_id
        #api_key = self.get_property("api_key").value

        #client = Gpt4Free(api_key=api_key)
        client = Gpt4Free()
        resp = client.chat.completions.create(
            model=model_id,
            messages=messages.copy(),
            tools=tools,
            tool_choice=options.get("tool_choice", None),
            stream=stream,
        )

        if stream:
            for chunk in resp:
                response.stream({
                        "choices": [{
                            "delta": {
                                "role": chunk.choices[0].delta.role,
                                "content": chunk.choices[0].delta.content
                            }
                        }]
                    })
            response.finish()
            return
        else:
            json_resp = json.loads(resp.model_dump_json())
            return json_resp
    
class Gpt4FreeInlineCompletionModel(InlineCompletionModel):
    def __init__(self, provider, model_id):
        super().__init__(provider)
        self._provider = provider
        self.model_id = model_id
        self._properties = [
            #LLMProviderProperty("api_key", "API key", "API key", "", True),
            LLMProviderProperty("context_window", "Context window", "Context window length", "", True),
        ]

    @property
    def id(self) -> str:
        return self.model_id
    
    @property
    def name(self) -> str:
        return self.model_id
    
    @property
    def context_window(self) -> int:
        try:
            context_window_prop = self.get_property("context_window")
            if context_window_prop is not None:
                context_window = int(context_window_prop.value)
            return context_window
        except:
            return DEFAULT_CONTEXT_WINDOW

    def inline_completions(self, prefix, suffix, language, filename, context: CompletionContext, cancel_token: CancelToken) -> str:
        model_id = self.model_id
        #api_key = self.get_property("api_key").value

        #client = Gpt4Free(api_key=api_key)
        client = Gpt4Free()
        resp = client.completions.create(
            model=model_id,
            prompt=prefix,
            suffix=suffix,
            stream=False,
        )

        return resp.choices[0].text

class Gpt4FreeLLMProvider(LLMProvider):
    def __init__(self):
        super().__init__()
        for provider in __providers__:
            if provider.working==True:
                if (hasattr(provider, 'models') and provider.needs_auth==False):
                    models = provider.models
                    for model in models:
                         if model not in LIST_MODELS:
                            m = Gpt4FreeChatModel(provider=self,model_id=model)
                            LIST_MODELS.append(m)
                            mi = Gpt4FreeInlineCompletionModel(provider=self,model_id=model)
                            LIST_INLINE_MODELS.append(mi)
        LIST_MODELS.sort(key=lambda x: x.model_id)
        LIST_INLINE_MODELS.sort(key=lambda x: x.model_id)
        self._chat_models = LIST_MODELS
        self._inline_completion_models = LIST_INLINE_MODELS

    @property
    def id(self) -> str:
        return "gpt4free-compatible"
    
    @property
    def name(self) -> str:
        return "Gpt4Free Compatible"

    @property
    def chat_models(self) -> list[ChatModel]:
        return self._chat_models
    
    @property
    def inline_completion_models(self) -> list[InlineCompletionModel]:
        return self._inline_completion_models
    
    @property
    def embedding_models(self) -> list[EmbeddingModel]:
        return []
