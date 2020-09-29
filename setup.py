'''lambdadata - a collect of Data Science Functions'''

import setuptools

REQUIRED = [
    "numpy",
    "pandas",
    "sklearn"
]

with open("README.MD", "r") as fh:
    LONG_DESCRIPTION = fh.read()

setuptools.setup(
    name='lambdata_t1',
    version='0.0.3',
    author="ttped",
    description="A collection of data science functions",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url='https://github.com/ttped/lambdata_trevor',
    packages=setuptools.find_packages(),
    install_requires=REQUIRED,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ],
    python_requires=">=3.6",
)
