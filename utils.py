#!/usr/bin/env python3
"""
Common utilities for handling context construction including RAG and verbose instruction.
"""

import os
from typing import Optional, Tuple


def get_retrieved_context(config_path: str, topk: int = 4, file_name: str = "retrieved_chunk_size_512_chunk_overlap_20_topk_4_embed_bge-large-en-v1.5.txt") -> Optional[str]:
    """Get retrieved context from RAG file."""
    context_path = os.path.join(os.path.dirname(config_path), file_name)
    if os.path.exists(context_path):
        with open(context_path, "r", encoding="utf-8") as f:
            context = f.read().strip()
        if context.strip() == "":
            return ''
        splits = context.split("Documentation Source:")
        if len(splits) > topk + 1:  # First split is empty
            return "Documentation Source:".join(splits[:topk + 1])
        return context
    return ''


def build_additional_contexts(
    task_config: dict,
    example_dir: str,
    use_rag: bool = False,
    use_verbose_instruction: bool = False,
    rag_topk: int = 4,
    rag_filename: str = "retrieved_chunk_size_512_chunk_overlap_20_topk_4_embed_bge-large-en-v1.5.txt"
) -> Tuple[Optional[str], Optional[str]]:
    """
    Build separate RAG context and verbose instruction content based on task config tags.
    This function is for frameworks that handle RAG and verbose instruction separately.
    """
    # Get task tags
    tags = task_config.get('tags', [])
    
    if use_rag:
        config_file_path = os.path.join(example_dir, f"{os.path.basename(example_dir)}.json")
        rag_context = get_retrieved_context(config_file_path, rag_topk, rag_filename)
        rag_context = f"\n\nWe also retrieve relevant documentation from the web to help you with the task:\n{rag_context}"
    else:
        rag_context = ''

    if use_verbose_instruction:
        verbose_instruction_path = os.path.join(example_dir, 'verbose_instruction.txt')
        with open(verbose_instruction_path, 'r', encoding='utf-8', errors='ignore') as f:
            verbose_instruction = f.read().strip()

        if 'abstract' in tags:
            verbose_content = f"\n\nHere is an abstract instruction for completing the task:\n{verbose_instruction}"
        else:
            verbose_content = f"\n\nHere is a step-by-step tutorial from an expert instructing you how to complete it:\n{verbose_instruction}"
            rag_context = ''
    else:
        verbose_content = ''
            
    return verbose_content+rag_context