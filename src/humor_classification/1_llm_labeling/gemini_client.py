"""
Gemini API Client with Structured Output using LangChain
"""

import os
import time
from typing import Optional, Callable
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm
from prompts import PROMPT_TEMPLATE

load_dotenv()

# Pricing per 1M tokens (USD) - Dec 2024
MODEL_PRICING = {
    # Gemini 2.0
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
    # Gemini 2.5
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    # Gemini 3 (preview)
    "gemini-3-flash-preview": {"input": 0.50, "output": 3.00},
    "gemini-3-pro-preview": {"input": 2.00, "output": 12.00},
    # Legacy
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-flash-8b": {"input": 0.0375, "output": 0.15},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
}


class HumorClassification(BaseModel):
    """Structured output for humor classification"""

    humor_type: str = Field(
        description="Primary humor type: affiliative, sexual, offensive, irony_satire, absurdist, or dark"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0"
    )
    explanation: str = Field(
        description="Brief explanation (1-2 sentences) of why this humor type was chosen"
    )


class GeminiClient:
    """Client for Gemini API with structured output via LangChain"""

    def __init__(
        self, model_name: str = "gemini-2.5-flash-lite", temperature: float = 0.1
    ):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found")

        self.model_name = model_name
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        # Create prompt template
        self.prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

        # Create LLM with structured output (return raw to get token usage)
        self.llm = ChatGoogleGenerativeAI(
            model=model_name, temperature=temperature, google_api_key=self.api_key
        )
        self.structured_llm = self.llm.with_structured_output(
            HumorClassification, include_raw=True
        )

        # Create chain: prompt | llm
        self.chain = self.prompt | self.structured_llm

        print(f"Initialized chain: prompt | {model_name} | HumorClassification")

    def _process_results(
        self, samples: list, inputs: list, batch_results: list
    ) -> list:
        """Helper to process batch results and track tokens. Returns list of (sample, input, result_dict)"""
        processed = []
        for sample, inp, raw_result in zip(samples, inputs, batch_results):
            result = raw_result["parsed"]
            raw_msg = raw_result.get("raw")

            # Track tokens
            if (
                raw_msg
                and hasattr(raw_msg, "usage_metadata")
                and raw_msg.usage_metadata
            ):
                usage = raw_msg.usage_metadata
                self.total_input_tokens += usage.get("input_tokens", 0)
                self.total_output_tokens += usage.get("output_tokens", 0)

            result_dict = result.model_dump()
            result_dict.update(
                {
                    "id": sample.get("id"),
                    "caption": sample["caption"],
                    "dataset": sample.get("dataset", ""),
                    "image_id": sample.get("image_id", ""),
                    "contest_number": sample.get("contest_number", ""),
                }
            )
            processed.append((sample, inp, result_dict))
        return processed

    def classify_batch(
        self,
        dataloader,
        checkpoint_fn: Optional[Callable] = None,
        checkpoint_every: int = 100,
        max_retries: int = 5,
        retry_delay: int = 15,
    ):
        """Classify samples using chain.batch() with retry logic"""
        results, processed, failed = [], 0, 0
        original_order = []  # Track original sample IDs for order verification

        # Use leave=True to keep progress bar visible and avoid multiple bars
        progress_bar = tqdm(
            dataloader, desc="Processing Batches", unit="batch", leave=True
        )

        for i, (batch, inputs) in enumerate(progress_bar):
            batch_results_list = []

            # Track original order for this batch
            batch_ids = [
                sample.get("id", f"batch{i}_idx{j}") for j, sample in enumerate(batch)
            ]
            original_order.extend(batch_ids)

            # Initial batch processing with retry
            for attempt in range(max_retries):
                try:
                    batch_results = self.chain.batch(inputs)
                    batch_results_list = self._process_results(
                        batch, inputs, batch_results
                    )
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(
                            f"\nWARNING: Batch error (attempt {attempt + 1}/{max_retries}): {e}"
                        )
                        print(f"   Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        print(
                            f"\nERROR: Batch failed after {max_retries} attempts: {e}"
                        )
                        failed += len(batch)
                        # Add placeholder results to maintain order
                        for sample in batch:
                            placeholder = {
                                "id": sample.get("id"),
                                "caption": sample["caption"],
                                "humor_type": "FAILED",
                                "confidence": 0.0,
                                "explanation": "Batch processing failed",
                                "dataset": sample.get("dataset", ""),
                                "image_id": sample.get("image_id", ""),
                                "contest_number": sample.get("contest_number", ""),
                            }
                            results.append(placeholder)
                            processed += 1
                        break

            if not batch_results_list:  # Batch completely failed
                continue

            for _, _, result_dict in batch_results_list:
                results.append(result_dict)
                processed += 1

            if checkpoint_fn and (i + 1) % checkpoint_every == 0:
                checkpoint_fn(results)

            time.sleep(0.3)

        # Close progress bar properly
        progress_bar.close()

        # Order verification check
        result_ids = [
            r.get("id", "UNKNOWN") for r in results if r.get("humor_type") != "FAILED"
        ]
        original_ids = [
            oid for oid in original_order if oid in [r.get("id") for r in results]
        ]

        if len(result_ids) > 0 and result_ids == original_ids[: len(result_ids)]:
            print(f"Order verification: PASSED ({len(result_ids)} samples)")
        elif len(result_ids) > 0:
            print(f"Order verification: FAILED - order mismatch detected")

        return results, processed, failed

    def get_usage(self) -> dict:
        """Return aggregated token usage and cost"""
        pricing = MODEL_PRICING.get(self.model_name, {"input": 0.075, "output": 0.30})
        input_cost = (self.total_input_tokens / 1_000_000) * pricing["input"]
        output_cost = (self.total_output_tokens / 1_000_000) * pricing["output"]

        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost,
            "model": self.model_name,
        }
