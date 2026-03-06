from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="ai-memory-system",
    version="0.1.0",
    author="PranayMahendrakar",
    description="Persistent AI Memory System for LLMs — long-term memory, conversation history, user preference learning, and vector memory.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PranayMahendrakar/ai-memory-system",
    packages=find_packages(exclude=["tests*", "examples*", "docs*"]),
    python_requires=">=3.9",
    install_requires=[
        # Zero mandatory dependencies — pure Python stdlib only.
        # Optional extras below.
    ],
    extras_require={
        "openai": ["openai>=1.0.0"],
        "sentence-transformers": ["sentence-transformers>=2.0.0"],
        "faiss": ["faiss-cpu>=1.7.0"],
        "numpy": ["numpy>=1.21.0"],
        "all": [
            "openai>=1.0.0",
            "sentence-transformers>=2.0.0",
            "numpy>=1.21.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
            "ruff>=0.1.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Typing :: Typed",
    ],
    keywords=[
        "llm", "memory", "ai", "chatbot", "vector-search",
        "conversation-history", "user-preferences", "embeddings",
        "langchain", "openai", "persistent-memory",
    ],
    project_urls={
        "Bug Tracker": "https://github.com/PranayMahendrakar/ai-memory-system/issues",
        "Documentation": "https://github.com/PranayMahendrakar/ai-memory-system/blob/main/README.md",
        "Source Code": "https://github.com/PranayMahendrakar/ai-memory-system",
    },
)
