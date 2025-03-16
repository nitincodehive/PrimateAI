# PrimateAI
A project to train an AI on primate behaviors and predict outcomes of different behavioral scenarios.

## Goal
Analyze and predict primate behavior outcomes based on contextual factors like behavior type, target, location, and social context.

## Features
- Predicts whether a primate behavior will result in positive, neutral, or negative outcomes
- Interactive command-line interface for behavior predictions
- Supports various primate behaviors, targets, locations, and social contexts
- Displays existing data combinations for reference

## Files
- `primate_data.csv`: Dataset containing primate behavior scenarios and outcomes
- `train_model.py`: Script to train the model and provide interactive predictions

## Usage
1. Run the script: `python train_model.py`
2. Enter behavior details when prompted (behavior, target, location, social context)
3. View the predicted outcome (positive, neutral, negative)

## Commands
- `help`: Display all valid options for behaviors, targets, locations, and social contexts
- `data`: View all existing data combinations in the dataset
- `quit`: Exit the program

## Data Structure
The model uses the following features:
- **Behavior**: Actions performed by primates (e.g., groom, fight, forage)
- **Target**: Object or subject of the behavior (e.g., food, mate, rival)
- **Location**: Where the behavior occurs (e.g., ground, tree, nest)
- **Social Context**: Social setting of the behavior (e.g., group, pair, alone)

## Outcome Values
- 1: Positive outcome
- 0: Neutral outcome
- -1: Negative outcome