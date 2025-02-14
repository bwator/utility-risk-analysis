from setuptools import setup, find_packages

setup(
    name='utility-risk-analysis',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'streamlit',
        'pandas',
        'numpy',
        'plotly',
        'scikit-learn',
        'matplotlib',
        'seaborn'
    ],
    author='Your Name',
    description='Utility Customer Risk Analysis Dashboard',
)