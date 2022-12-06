# README

I had to install HomeBrew https://brew.sh/ and the following dependencies:

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

I use `pipenv` for my Python stuff. Find it easier to use. https://pipenv.pypa.io/en/latest/install/

```zsh
pip install --user pipenv
```

Then create the virtual environment and install dependencies:

```zsh
pipenv install
```

Run the code:

```ash
pipenv shell
python dataset_generation.py
python feature_transformation.py
python logistic_regression.py
python convolutional_neural_network.py
```

Or using Jupyter:

```zsh
pipenv run jupyter lab
```

A browser will be opened, and you can open any .ipynb file and run it.


Lint and format code:

```zsh
pipenv shell

isort --profile=black *.py
black *.py
pylint -E *.py

nbqa isort --profile=black *.ipynb
nbqa black *.ipynb
nbqa pylint *.ipynb

jupyter nbconvert --clear-output --inplace *.ipynb
```