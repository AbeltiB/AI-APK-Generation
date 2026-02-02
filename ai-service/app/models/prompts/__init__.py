"""
Prompt templates for AI service.
"""
from .templates import (
    PromptTemplate,
    PromptLibrary,
    PromptType,
    PromptVersion
)

# Create an instance of PromptLibrary
prompts = PromptLibrary()

__all__ = [
    'PromptTemplate',
    'PromptLibrary',
    'PromptType',
    'PromptVersion',
    'prompts'  
]