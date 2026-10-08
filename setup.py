"""
setup.py / pyproject-style setup for the mini-engine package.
Install in editable mode with: pip install -e .
"""
from setuptools import setup, find_packages

setup(
    name="mini-llm-engine",
    version="0.1.0",
    description="A mini LLM inference engine with continuous batching and paged KV cache",
    packages=find_packages(exclude=["tests*", "bench*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.44.0,<4.47.0",
        "huggingface-hub>=0.20.0",
        "fastapi>=0.110.0",
        "uvicorn[standard]>=0.27.0",
        "safetensors>=0.4.0",
        "pydantic>=2.0.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "httpx>=0.25.0"],
    },
)
