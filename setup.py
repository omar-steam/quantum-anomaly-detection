
from setuptools import setup, find_packages

setup(
    name="quantum-network-anomaly",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'qiskit>=0.44.1',
        'qiskit-machine-learning>=0.6.1',
        'qiskit-aer>=0.12.2',
        'pandas>=2.1.1',
        'numpy>=1.24.3',
        'scikit-learn>=1.3.0',
        'scipy>=1.11.3',
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A quantum machine learning approach for network anomaly detection",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/quantum-network-anomaly",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
