# GLC24
Geolifeclef 2024 competition

Competition: [https://www.kaggle.com/competitions/geolifeclef-2024/overview](https://www.kaggle.com/competitions/geolifeclef-2024/overview)

Next steps;
- Download data from seafile as well?
- 
Initial thoughts:
- Use nearby (0.1-1 deg) species of same/similar LC as hard mask (to prevent FP)? Might be more efficient than species co-occurrence mask due to large number of species.
- Can we use supervised CL on PO data? Different input samples of same species should have minimal-distance embeddings. It stated somewhere (CHECK) that many locations only have a single sample: So can we reverse the data structure? For each species, what are all the locations. These should be mapped to the same space. During training, grab 2 random species, 16 samples each (for example .. prevent trivial solutions of all input to same embedding). Then CL of same species. But not exactly a perfect approach; co-occurence would be a better indicator for embedding than single occurence. 