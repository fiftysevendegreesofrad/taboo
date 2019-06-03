# taboo
Data mining for what isn't said

Still at prototype stage - read the source to figure out what's going on!

* `taboo_make_matrix.py` converts a text corpus to cross-correlation matrix, word list and story count
* `taboo_from_matrix.py` analyzes output of taboo_make_matrix. It forms a model for all words A, B to predict `P(A present in text|B present in text)`. It then looks for outliers where this model over or underpredicts `P(A|B)`. It would likely work better if we used topics rather than words.
* `merge_matrices.py` is only needed if you ran taboo_make_matrix as multiple jobs in a PBS batch. It merges multiple matrices into one. 

