from setuptools import setup, find_packages

setup(
    name="nonlinear-timeseries-analyzer",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive toolkit for nonlinear time series analysis",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/nonlinear-time-series-analysis",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "networkx>=2.6.0",
    ],
)
