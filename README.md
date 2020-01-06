# Oxford MSc Dissertation - Grade: 81% (Distinction)
## Automatic Patent Searching via Extreme Multi-Label Classification

My Oxford MSc thesis aimed to reduce the workload of patent attorneys
interested in ascertaining the patentability of a new drug. I attempted
to do this by building a classification model which when given a compound
of interest as its input would output the US drug patents most likely
to cover it.

The data used in the project were extracted from US drug patents by NextMove
and are publicly available. Rather than using ALL US drug patents, 10% of the
all molecules were randomly selected resulting in a dataset of 100,000 unique
molecular compounds, 15,000 patents (the labels), and 450,000 distinct molecular
substructures (each one a binary feature vector).

The model used was an ensemble of tree-based learners which were stacked using
a majority voting procedure. The reason for using tree-based learners was due
to training times - the data matrix while extremely large (100,000 x 450,000)
was also extremely sparse. The training procedures for Random Forest and
Extremely Random Trees models were able to exploit this sparsity (theoretical
justifications are posited in the thesis document) resulting in training times
of the order of minutes when trained on large clusters compared to hours for
even simple XGBoost and shallow Neural Network models. Classification accuracy
was also far higher in the tree-based models.

The final model consisted of three Random Forest and Four Extremely Random Trees
models, stacked using majority voting. Hyperparameters for these seven models were
tuned using Bayesian Optimisation.

main.py is the full pipeline and implements the Bayesian optimisation, with results
being saved in the Reports folder.

The data provided is a toy example - the full data set used in the project resulted
in base models which used 400+ GB of memory and hence required a large cluster to run.
Running the main.py script with the data provided should use approximately 4 GB of 
memory and so can be run on most desktop PC's.

Dissertation.pdf is the final submitted document.


