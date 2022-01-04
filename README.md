Solution for kaggle [spooky-author-identification](https://www.kaggle.com/c/spooky-author-identification) competition

Pure R, tested on ubuntu 18.04 / 20.04

To reproduce it, open "solution.Rmd" and *Knit* it. You'll get the same results as in the "solution.html" report (NN accuracy/loss may vary slightly)

During execution it may ask to install additional vocabularies, such as [AFINN](https://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=6010) and [NRC](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm)

XGBoost chunk is annotated with ```eval = F``` to suppress nasty deprecation warnings. To get a final submission you need to remove it.

All chunks are cached, so in case of any errors (e.g. missed package) execution can continue from the broken chunk.

This solution also contains many ideas and approaches taken from Kaggle, so many thanks to the other kagglers for the great stuff!
