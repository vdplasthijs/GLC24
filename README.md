# GLC24
Geolifeclef 2024 competition

Competition: [https://www.kaggle.com/competitions/geolifeclef-2024/overview](https://www.kaggle.com/competitions/geolifeclef-2024/overview)

Next steps;
- Download data from seafile as well?
- Look at example notebooks to understand how to index data etc.
- 
Initial thoughts:
- How many locations, how many species per location, how many species, for PA train/PO train/PA test.?
- Use prior info; get list of species per country, only allow those species to be predicted (per country).
- Create species co-occurence matrix (from both PA and PO I think, though most PO might be single obs).
- Can we use supervised CL on PO data? Different input samples of same species should have minimal-distance embeddings. It stated somewhere (CHECK) that many locations only have a single sample: So can we reverse the data structure? For each species, what are all the locations. These should be mapped to the same space. During training, grab 2 random species, 16 samples each (for example .. prevent trivial solutions of all input to same embedding). Then CL of same species. But not exactly a perfect approach; co-occurence would be a better indicator for embedding than single occurence. 
