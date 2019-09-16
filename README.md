## MSc Dissertation
# Automatic Patent Searching Via Extreme Multi-Label Classification


This is some code for the model stacking approach I used in my Oxford MSc Thesis.
It performs a Bayesian optimisation of the hyperparameters of the base learners
which are fed to the level two model stacker.

main.py is the full pipeline and implements the Bayesian optimisation.

The data used is a toy example - the full data set used in the project resulted
in base models which used 400+ GB of memory. Running the main.py script with the
data provided should use approximately 4 GB of memory and so can be run on most
desktop PC's.

Dissertation.pdf is the submitted document.


