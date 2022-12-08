# Credit Card Fraud Detection Sample

This project is based on 
[Reproducible Machine Learning for Credit Card Fraud Detection - Practical Handbook](https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook)

## How to run

To run this project in MacOS, you need to get some dependencies installed with [HomeBrew](https://brew.sh/).
First install `brew`, and then install the following packages:

```zsh
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install \
    cython \
    freetype \
    git
    graphviz \
    libomp \
    pkg-config
```

This project use [Pipenv](https://pipenv.pypa.io/en/latest/install/) to manage dependencies.
Install it using `pip`:

```zsh
pip install --user pipenv
```

Then, create a virtual environment and install dependencies:

```zsh
pipenv install
```

The code can be run from a terminal. To run the code, first activate the virtual environment:

```zsh
pipenv shell
```

And then you can each step, one by one:

```
python dataset_generation.py
python feature_transformation.py
python logistic_regression.py
python convolutional_neural_network.py
```

Or using Jupyter Lab. A browser will be opened, and you can open `notebook.ipynb` and run it:

```zsh
pipenv run jupyter lab
```

The notebook can be also imported in [Google Collab](https://colab.research.google.com/).

## Development

```zsh
pipenv shell

# Lint and format code:
isort --profile=black *.py
black *.py
pylint -E *.py

# Clear the Jupyter Notebook output:
jupyter nbconvert --clear-output --inplace *.ipynb
```
