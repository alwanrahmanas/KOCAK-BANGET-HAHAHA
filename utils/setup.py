"""
Setup configuration for BPJS Fraud Detection package
"""

from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='bpjs-fraud-detection',
    version='2.0.0',
    author='BPJS Fraud Detection Team',
    description='ML-based fraud detection system for BPJS healthcare claims',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Healthcare Industry',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    entry_points={
        'console_scripts': [
            'bpjs-fraud=bpjs_fraud_detection.main:main',
        ],
    },
)
