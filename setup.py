from pathlib import Path

from setuptools import find_packages, setup


README = Path(__file__).with_name("README.md").read_text(encoding="utf-8")

setup(
    name="oscar-hallucination",
    version="0.1.0",
    description="OSCAR: Orchestrated Self-verification and Cross-path Refinement for hallucination detection and correction in diffusion language models",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Yash Shah, Abhijit Chakraborty, Naresh Kumar Devulapally, Vishnu Lokhande, Vivek Gupta",
    url="https://arxiv.org/abs/2604.01624",
    license="MIT",
    packages=find_packages(exclude=["tests*", "results*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.3.0",
        "transformers>=4.43.0",
        "datasets>=2.20.0",
        "scipy>=1.13.0",
        "scikit-learn>=1.5.0",
        "numpy>=1.26.0",
        "tqdm>=4.66.0",
        "accelerate>=0.31.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-cov>=5.0.0",
            "black>=24.0.0",
            "ruff>=0.4.0",
        ],
        "gpu": [
            "bitsandbytes>=0.43.0",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
)
