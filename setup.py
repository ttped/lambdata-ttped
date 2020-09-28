'''lambdadata - a collect of Data Science Functions'''

import setuptools

REQUIRED = [
    "numpy",
    "pandas",
    "datetime",
    "sklearn"
]

with open("README.MD", "r") as fh:
    LONG_DESCRIPTION = fh.read()

setuptools.setup(
    name='lambdata-ttped',
    version='0.0.1',
    author="ttped",
    description="A collect of data science functions",
    long_description=LONG_DESCRIPTION,
    long_description_content="text/markdown",
    url='https://github.com/ttped/lambdata_ttped',
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires=REQUIRED,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS independent'
    ]
)
