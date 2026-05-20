import setuptools
from pathlib import Path

project_dir = Path(__file__).parent
readme_path = project_dir / "docs" / "AtomicRAG_README.md"
fallback_readme = project_dir / "README.md"

with open(readme_path if readme_path.exists() else fallback_readme, "r", encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="atomicrag",
    version="0.1.0",
    author="Yanning Hou, Duanyang Yuan, Sihang Zhou, Xiaoshu Chen, Ke Liang, Siwei Wang, Xinwang Liu, Jian Huang",
    author_email="sihangjoe@gmail.com",
    description="Official implementation of AtomicRAG: Atom-Entity Graphs for Retrieval-Augmented Generation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/7HHHHH/AtomicRAG",
    project_urls={
        "Paper": "https://arxiv.org/abs/2604.20844",
        "Source": "https://github.com/7HHHHH/AtomicRAG",
    },
    package_dir={"": "."},
    packages=setuptools.find_packages(include=["atomicrag", "atomicrag.*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch==2.5.1",
        "transformers==4.45.2",
        "openai==1.92.1",
        "litellm==1.73.1",
        "gritlm==1.0.2",
        "networkx==3.4.2",
        "python_igraph==0.11.8",
        "tiktoken==0.7.0",
        "pydantic==2.10.4",
        "tenacity==8.5.0",
        "einops",
        "tqdm",
        "boto3",
    ]
)
