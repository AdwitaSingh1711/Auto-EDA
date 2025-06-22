from setuptools import setup, find_packages

setup(
    name = 'autoeda',
    version = '0.1.0',
    packages = find_packages(),
    install_requires = [
        'pandas',
        'numpy',
        'plotly',
        'loguru',
        'argparse',
        'ollama',
        'scikit-learn',
        'matplotlib',
        'seaborn',
    ],
    entry_points ={
        'console_scripts':[
            'autoeda=autoeda.autoeda:main'
        ]
    },

    author = 'Adwita Singh',
    author_email = 'adwita.s.at07@gmail.com',
    description = 'Automated Exploratory Data Analysis (EDA) Tool',
    long_description = open('README.md').read(),
    long_description_content_type ='text/markdown',
    url = 'https://github.com/AdwitaSingh1711/Auto-EDA',
    classifiers = [
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ], 
    python_requires = '3.11.1',
)