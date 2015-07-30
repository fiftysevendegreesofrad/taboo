# taboo
Data mining for what isn't said

taboo_make_matrix.py converts corpus to cross-correlation matrix, word list and story count
taboo_from_matrix.py analyzes output of taboo_make_matrix 

merge_matrices.py is only needed if you ran taboo_make_matrix as multiple jobs in a PBS batch.
It merges multiple matrices into one. 
If you're going to use it, read the source first, it probably needs adapting to your file naming scheme.