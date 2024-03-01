from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='XGeoML',
    version='0.1.5',
    author='Lingbo Liu',
    author_email='lingboliu@fas.harvard.edu',
    description='A ensemble framework for explainable geospatial machine Learning models',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/UrbanGISer/XGeoML",
    packages=find_packages(), # Corrected here
    license="MIT",
    install_requires=[
        'scikit-learn',
        'numpy',
        'scipy',
        'joblib',
        'pandas',
        'lime',
        'shap',
        'tqdm', 
    ],
)
