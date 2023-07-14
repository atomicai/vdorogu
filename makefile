LINE_WIDTH=122
NAME := $(shell python setup.py --name)
UNAME := $(shell uname -s)
ISORT_FLAGS=--line-width=${LINE_WIDTH} --profile black
FLAKE_FLAGS=--remove-unused-variables --ignore-init-module-imports --recursive
# "" is for multi-lang strings (comments, logs), '' is for everything else.
BLACK_FLAGS=--skip-string-normalization --line-length=${LINE_WIDTH}
PYTEST_FLAGS=-p no:warnings
export FLASK_APP=vdorogu.tdk.rise

install:
	pip install -e '.[all]'

init:
	pip install pre-commit==3.3.3
	pre-commit clean
	pre-commit install
  	# To check whole pipeline.
	# pre-commit run --all-files


format:
	isort ${NAME} external test
	black ${NAME} external test

run:
	flask run --host=0.0.0.0 --port=7777

test:
	pytest test ${PYTEST_FLAGS} --testmon --suppress-no-test-exit-code

test-all:
	pytest test ${PYTEST_FLAGS}

clean:
	rm -rf .ipynb_checkpoints
	rm -rf **/.ipynb_checkpoints
	rm -rf .pytest_cache
	rm -rf build
	rm -rf dist
	rm -rf downloads
	rm -rf wandb
	find . -name ".DS_Store" -print -delete
	rm -rf .cache
	pyclean .
