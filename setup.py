"""
Setup configuration for addGPT package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
description = (this_directory / "README.md").read_text()

setup(
    name="addGPT",
    version="1.0.0",
    description="Teaching transformers to add numbers - a minimal transformer implementation",
    description=description,
    description_content_type="text/markdown",
    url="https://github.com/mbano/addGPT",
    packages=find_packages(),
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "optuna>=3.0.0",
        "tensorboard>=2.13.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "huggingface-hub>=0.16.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "addGPT-train=train:main",
            "addGPT-demo=demo:main",
            "addGPT-download=download_checkpoint:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.yaml", "config/**/*.yaml"],
    },
    keywords="transformer deep-learning pytorch addition algorithmic-learning",
    project_urls={
        "Source": "https://github.com/mbano/addGPT",
        "Documentation": "https://github.com/mbano/addGPT#readme",
    },
)
