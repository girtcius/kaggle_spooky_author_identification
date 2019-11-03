require(tidyverse)
require(tidytext)
require(textstem)
require(qdap)
require(caret)
require(widyr)
require(broom)
require(keras)
require(gridExtra)
require(plotly)
require(scales)
require(ggcorrplot)
require(RDRPOSTagger)
require(parallel)
require(gmodels)

numCores <- detectCores()

train <- read_csv('data/train.csv') %>% rename(ID = id)
test <- read_csv('data/test.csv') %>% rename(ID = id)

head(train)
glimpse(train)
summary(train)

head(test)
glimpse(test)
summary(test)

colSums(is.na(train))
colSums(is.na(test))

table(train$author) %>% prop.table()

wordProportion <- train %>% 
    unnest_tokens(word, text, token = "ngrams", n=1) %>%
    anti_join(stop_words, by = "word") %>%
    count(author, word) %>%
    group_by(author)  %>%
    mutate(proportion = n / sum(n)) %>% 
    select(-n) %>%
    spread(author, proportion)

wordProportion %>% 
    filter(!is.na(EAP) & !is.na(HPL)) %>%
    ggplot(aes(x = EAP, y = HPL, color = abs(EAP - HPL))) +
    geom_abline(color = "gray40", lty = 2) +
    geom_jitter(alpha = 0.1, size = 2, width = 0.3, height = 0.3) +
    geom_text(aes(label = word), check_overlap = TRUE, vjust = 1.5) +
    scale_x_log10(labels = percent_format()) +
    scale_y_log10(labels = percent_format()) +
    scale_color_distiller(palette = "PuRd") +
    theme(legend.position="none") +
    labs(x = "Edgar Allan Poe", y = "Howard Phillips Lovecraft")

cor.test(data = wordProportion, ~ EAP + HPL)

# EAP vs MWS word proportion plot 
wordProportion %>% 
    filter(!is.na(EAP) & !is.na(MWS)) %>%
    ggplot(aes(x = EAP, y = MWS, color = abs(EAP - MWS))) +
    geom_abline(color = "gray40", lty = 2) +
    geom_jitter(alpha = 0.1, size = 2, width = 0.3, height = 0.3) +
    geom_text(aes(label = word), check_overlap = TRUE, vjust = 1.5) +
    scale_x_log10(labels = percent_format()) +
    scale_y_log10(labels = percent_format()) +
    scale_color_distiller(palette = "BuGn") +
    theme(legend.position="none") +
    labs(x = "Edgar Allan Poe", y = "Mary Wollstonecraft Shelley")

cor.test(data = wordProportion, ~ EAP + MWS)

# HPL vs MWS word proportion plot 
wordProportion %>% 
    filter(!is.na(HPL) & !is.na(MWS)) %>%
    ggplot(aes(x = HPL, y = MWS, color = abs(HPL - MWS))) +
    geom_abline(color = "gray40", lty = 2) +
    geom_jitter(alpha = 0.1, size = 2, width = 0.3, height = 0.3) +
    geom_text(aes(label = word), check_overlap = TRUE, vjust = 1.5) +
    scale_x_log10(labels = percent_format()) +
    scale_y_log10(labels = percent_format()) +
    scale_color_distiller(palette = "RdGy") +
    theme(legend.position="none") +
    labs(x = "Howard Phillips Lovecraft", y = "Mary Wollstonecraft Shelley")

cor.test(data = wordProportion, ~ HPL + MWS)

# Basic stylometric features
train_stylo <- train %>% select(ID) %>%
    mutate(word_count = str_count(train$text, '\\w+'),
           syll_count = as.vector(syllable_sum(
               iconv(train$text, to='ASCII//TRANSLIT'), parallel = T)),  # syllable count
           nsyll_per_word = as.vector(syllable_sum(
               iconv(train$text, to='ASCII//TRANSLIT'))/word_count),  # number of syllables per word
           nchar = nchar(train$text),                               # character count
           nchar_per_word = nchar/word_count,                       # number of characters per word
           tc_count = str_count(train$text, "[A-Z][a-z]+"),         # title case words count
           uc_count = str_count(train$text, "[A-Z][A-Z]+"),         # upper case words count
           punctuation_count = str_count(train$text, "[:punct:]"))  

test_stylo <- test %>% select(ID) %>%
    mutate(word_count = str_count(test$text, '\\w+'),
           syll_count = as.vector(syllable_sum(
               iconv(test$text, to='ASCII//TRANSLIT'), parallel = T)),
           nsyll_per_word = as.vector(syllable_sum(iconv(test$text, to='ASCII//TRANSLIT'))/word_count),
           nchar = nchar(test$text),
           nchar_per_word = nchar/word_count,
           tc_count = str_count(test$text, "[A-Z][a-z]+"),
           uc_count = str_count(test$text, "[A-Z][A-Z]+"),
           punctuation_count = str_count(test$text, "[:punct:]"))


train_tmp <- train %>% unnest_tokens(term, text, token = "ngrams", n=1) %>%
    mutate(stop_word_ind = as.integer(term %in% stop_words$word),
           # syll_count = as.vector(syllable_sum(
           # iconv(.$term, to='ASCII//TRANSLIT'), parallel = T)),
           word_length = nchar(term)) %>%
    group_by(ID) %>%
    summarise(nStopWord = sum(stop_word_ind),         # stopwords count
              nUniqueWord = n_distinct(term),         # unique words count
              avg_word_length = mean(word_length),    # average count of characters in word
              ngt6lWord = sum(word_length > 6),       # count of words with more than 6 characters
              # n1SylWord = sum(syll_count == 1),       # count of words with only one syllable
              # nPolySylWord = sum(syll_count > 2)
              )     # count of words with more than 2 syllables

test_tmp <- test %>% unnest_tokens(term, text, token = "ngrams", n=1) %>%
    mutate(stop_word_ind = as.integer(term %in% stop_words$word),
           # syll_count = as.vector(syllable_sum(
           # iconv(.$term, to='ASCII//TRANSLIT'))),
           word_length = nchar(term)) %>%
    group_by(ID) %>%
    summarise(nStopWord = sum(stop_word_ind),
              nUniqueWord = n_distinct(term),
              avg_word_length = mean(word_length),
              ngt6lWord = sum(word_length > 6),
              # n1SylWord = sum(syll_count == 1),
              # nPolySylWord = sum(syll_count > 2)
              )

train_stylo <- train_stylo %>% 
    left_join(train_tmp, by = "ID") %>%
    mutate(unique_r = nUniqueWord/word_count,
           w_p = word_count - punctuation_count,
           w_p_r = w_p/word_count,
           stop_r = nStopWord/word_count,
           w_p_stop = w_p - nStopWord,
           w_p_stop_r = w_p_stop/word_count,
           num_words_upper_r = uc_count/word_count,
           num_words_title_r = tc_count/word_count)

test_stylo <- test_stylo %>% 
    left_join(test_tmp, by = "ID") %>%
    mutate(unique_r = nUniqueWord/word_count,
           w_p = word_count - punctuation_count,
           w_p_r = w_p/word_count,
           stop_r = nStopWord/word_count,
           w_p_stop = w_p - nStopWord,
           w_p_stop_r = w_p_stop/word_count,
           num_words_upper_r = uc_count/word_count,
           num_words_title_r = tc_count/word_count)

# Features based on verb types and qualifiers
thought_verbs <- c("analyze", "apprehend", "assume", "believe", "calculate", "cerebrate", "cogitate",
                   "comprehend", "conceive", "concentrate", "conceptualize", "conclude", "consider",
                   "construe", "contemplate", "deduce", "deem", "delibrate", "desire", "diagnose",
                   "doubt", "envisage", "envision", "evaluate", "excogitate", "extrapolate", "fantasize",
                   "forget", "forgive", "formulate", "hate", "hypothesize", "imagine", "infer", 
                   "intellectualize", "intrigue", "guess", "introspect", "judge", "know", "love", 
                   "lucubrate", "marvel", "meditate", "note", "notice", "opine", "perpend", "philosophize",
                   "ponder", "question", "ratiocinate", "rationalize", "realize", "reason", "recollect", 
                   "reflect", "remember", "reminisce", "retrospect", "ruminate", "sense", "speculate",
                   "stew", "strategize", "suppose", "suspect", "syllogize", "theorize", "think", 
                   "understand", "visualize", "want", "weigh", "wonder")

loud_verbs <- c("cry", "exclaim", "shout", "roar", "scream", "shriek", "vociferated", "bawl",
                "call", "ejaculate", "retort", "proclaim", "announce", "protest", "accost", "declare")

neutral_verbs <- c("say", "reply", "observe", "rejoin", "ask", "answer", "return", "repeat", "remark",
                   "enquire", "respond", "suggest", "explain", "utter", "mention")

quiet_verbs <- c("whisper", "murmur", "sigh", "grumble", "mumble", "mutter", "whimper", "hush", "falter",
                 "stammer", "tremble", "gasp", "shudder")

qualifiers <- c("very", "too", "so", "quite", "rather", "little", "pretty", "somewhat", "various", "almost", 
                "much", "just", "indeed", "still", "even", "a lot", "kind of", "sort of")

train_tmp <- train %>% unnest_tokens(term, text, token = "ngrams", n=1) %>% 
    bind_rows(train %>% unnest_tokens(term, text, token = "ngrams", n=2)) %>%
    mutate(term = lemmatize_words(term),
           qualifier_ind = as.integer(term %in% qualifiers),
           thought_verbs_ind = as.integer(term %in% thought_verbs),
           loud_verbs_ind = as.integer(term %in% loud_verbs),
           neutral_verbs_ind = as.integer(term %in% neutral_verbs),
           quiet_verbs_ind = as.integer(term %in% quiet_verbs)) %>%
    group_by(ID) %>%
    summarise(qualifier_count = sum(qualifier_ind),
              thought_verbs_count = sum(thought_verbs_ind),
              loud_verbs_count = sum(loud_verbs_ind),
              neutral_verbs_count = sum(neutral_verbs_ind),
              quiet_verbs_count = sum(quiet_verbs_ind))

test_tmp <- test %>% unnest_tokens(term, text, token = "ngrams", n=1) %>% 
    bind_rows(test %>% unnest_tokens(term, text, token = "ngrams", n=2)) %>%
    mutate(term = lemmatize_words(term),
           qualifier_ind = as.integer(term %in% qualifiers),
           thought_verbs_ind = as.integer(term %in% thought_verbs),
           loud_verbs_ind = as.integer(term %in% loud_verbs),
           neutral_verbs_ind = as.integer(term %in% neutral_verbs),
           quiet_verbs_ind = as.integer(term %in% quiet_verbs)) %>%
    group_by(ID) %>%
    summarise(qualifier_count = sum(qualifier_ind),
              thought_verbs_count = sum(thought_verbs_ind),
              loud_verbs_count = sum(loud_verbs_ind),
              neutral_verbs_count = sum(neutral_verbs_ind),
              quiet_verbs_count = sum(quiet_verbs_ind))

train_stylo <- train_stylo %>% 
    left_join(train_tmp, by = "ID")

test_stylo <- test_stylo %>% 
    left_join(test_tmp, by = "ID")

stylo_plot <- train  %>% left_join(train_stylo, by = "ID") %>% select(-text,-ID) %>%
    group_by(author) %>%
    summarise_all(mean) %>%
    gather(feature, value, -author) %>%
    ggplot(aes(x = feature, y = value, color = author, fill = author, size = log10(value))) +
    geom_point(alpha = 0.6) +
    scale_y_log10() +
    labs(x = "Feature", y = "Mean Value by Author") +
    coord_flip()
ggplotly(stylo_plot, tooltip = c("x","y","fill"))

# Author specific unigrams
author_unigrams_tfidf <- train %>% unnest_tokens(term, text, token = "ngrams", n=1) %>%
    count(author, term) %>% bind_tf_idf(term, author, n)

# Author specific bigrams
author_bigrams_tfidf <- train %>% unnest_tokens(term, text, token = "ngrams", n=2) %>%
    count(author, term) %>% bind_tf_idf(term, author, n)

# Author specific trigrams
author_trigrams_tfidf <- train %>% unnest_tokens(term, text, token = "ngrams", n=3) %>%
    count(author, term) %>% bind_tf_idf(term, author, n)

# Author specific tetragrams
author_tetragrams_tfidf <- train %>% unnest_tokens(term, text, token = "ngrams", n=4) %>%
    count(author, term) %>% bind_tf_idf(term, author, n)

# Author specific character shingle bigrams
author_char_bigrams_tfidf <- train %>% 
    unnest_tokens(shingle, text, token = "character_shingles", n=2, strip_non_alphanum = FALSE) %>%
    count(author, shingle) %>% bind_tf_idf(shingle, author, n)

# Author specific character shingle trigrams
author_char_trigrams_tfidf <- train %>% 
    unnest_tokens(shingle, text, token = "character_shingles", n=3, strip_non_alphanum = FALSE) %>%
    count(author, shingle) %>% bind_tf_idf(shingle, author, n)

# Author specific character shingle tetragrams
author_char_tetragrams_tfidf <- train %>% 
    unnest_tokens(shingle, text, token = "character_shingles", n=4, strip_non_alphanum = FALSE) %>%
    count(author, shingle) %>% bind_tf_idf(shingle, author, n)

# Author specific character shingle pentagrams
author_char_pentagrams_tfidf <- train %>% 
    unnest_tokens(shingle, text, token = "character_shingles", n=5, strip_non_alphanum = FALSE) %>%
    count(author, shingle) %>% bind_tf_idf(shingle, author, n)

# Author only terms
EAP_only <- 
    unique(c(author_unigrams_tfidf %>% filter(idf == log(3) & author == "EAP" & n > 25) %>% .$term,
             author_bigrams_tfidf %>% filter(idf == log(3) & author == "EAP" & n > 25) %>% .$term,
             author_trigrams_tfidf %>% filter(idf == log(3) & author == "EAP" & n > 25) %>% .$term,
             author_tetragrams_tfidf %>% filter(idf == log(3) & author == "EAP" & n > 25) %>% .$term,
             author_char_bigrams_tfidf %>% filter(idf == log(3) & author == "EAP" & n > 25) %>% .$shingle,
             author_char_trigrams_tfidf %>% filter(idf == log(3) & author == "EAP" & n > 25) %>% .$shingle,
             author_char_tetragrams_tfidf %>% filter(idf == log(3) & author == "EAP" & n > 25) %>% .$shingle,
             author_char_pentagrams_tfidf %>% filter(idf == log(3) & author == "EAP" & n > 25) %>% .$shingle))

HPL_only <- 
    unique(c(author_unigrams_tfidf %>% filter(idf == log(3) & author == "HPL" & n > 25) %>% .$term,
             author_bigrams_tfidf %>% filter(idf == log(3) & author == "HPL" & n > 25) %>% .$term,
             author_trigrams_tfidf %>% filter(idf == log(3) & author == "HPL" & n > 25) %>% .$term,
             author_tetragrams_tfidf %>% filter(idf == log(3) & author == "HPL" & n > 25) %>% .$term,
             author_char_bigrams_tfidf %>% filter(idf == log(3) & author == "HPL" & n > 25) %>% .$shingle,
             author_char_trigrams_tfidf %>% filter(idf == log(3) & author == "HPL" & n > 25) %>% .$shingle,
             author_char_tetragrams_tfidf %>% filter(idf == log(3) & author == "HPL" & n > 25) %>% .$shingle,
             author_char_pentagrams_tfidf %>% filter(idf == log(3) & author == "HPL" & n > 25) %>% .$shingle))

MWS_only <- 
    unique(c(author_unigrams_tfidf %>% filter(idf == log(3) & author == "MWS" & n > 25) %>% .$term,
             author_bigrams_tfidf %>% filter(idf == log(3) & author == "MWS" & n > 25) %>% .$term,
             author_trigrams_tfidf %>% filter(idf == log(3) & author == "MWS" & n > 25) %>% .$term,
             author_tetragrams_tfidf %>% filter(idf == log(3) & author == "MWS" & n > 25) %>% .$term,
             author_char_bigrams_tfidf %>% filter(idf == log(3) & author == "MWS" & n > 25) %>% .$shingle,
             author_char_trigrams_tfidf %>% filter(idf == log(3) & author == "MWS" & n > 25) %>% .$shingle,
             author_char_tetragrams_tfidf %>% filter(idf == log(3) & author == "MWS" & n > 25) %>% .$shingle,
             author_char_pentagrams_tfidf %>% filter(idf == log(3) & author == "MWS" & n > 25) %>% .$shingle))

train_author_only <- train %>%
    unnest_tokens(term, text, token = "ngrams", n=1) %>% 
    bind_rows(train %>%
                  unnest_tokens(term, text, token = "ngrams", n=2)) %>%
    bind_rows(train %>%
                  unnest_tokens(term, text, token = "ngrams", n=3)) %>%
    bind_rows(train %>%
                  unnest_tokens(term, text, token = "ngrams", n=4)) %>%
    bind_rows(train %>% 
                  unnest_tokens(term, text, token = "character_shingles", n=2, strip_non_alphanum = FALSE)) %>%
    bind_rows(train %>%
                  unnest_tokens(term, text, token = "character_shingles", n=3, strip_non_alphanum = FALSE)) %>%
    bind_rows(train %>%
                  unnest_tokens(term, text, token = "character_shingles", n=4, strip_non_alphanum = FALSE)) %>%
    bind_rows(train %>%
                  unnest_tokens(term, text, token = "character_shingles", n=5, strip_non_alphanum = FALSE)) %>%
    mutate(EAP_only_ind = as.integer(term %in% EAP_only),
           HPL_only_ind = as.integer(term %in% HPL_only),
           MWS_only_ind = as.integer(term %in% MWS_only)) %>%
    group_by(ID) %>%
    summarise(EAP_only_count = sum(EAP_only_ind),
              HPL_only_count = sum(HPL_only_ind),
              MWS_only_count = sum(MWS_only_ind))

test_author_only <- test %>%
    unnest_tokens(term, text, token = "ngrams", n=1) %>% 
    bind_rows(test %>%
                  unnest_tokens(term, text, token = "ngrams", n=2)) %>%
    bind_rows(test %>%
                  unnest_tokens(term, text, token = "ngrams", n=3)) %>%
    bind_rows(test %>%
                  unnest_tokens(term, text, token = "ngrams", n=4)) %>%
    bind_rows(test %>% 
                  unnest_tokens(term, text, token = "character_shingles", n=2, strip_non_alphanum = FALSE)) %>%
    bind_rows(test %>%
                  unnest_tokens(term, text, token = "character_shingles", n=3, strip_non_alphanum = FALSE)) %>%
    bind_rows(test %>%
                  unnest_tokens(term, text, token = "character_shingles", n=4, strip_non_alphanum = FALSE)) %>%
    bind_rows(test %>%
                  unnest_tokens(term, text, token = "character_shingles", n=5, strip_non_alphanum = FALSE)) %>%
    mutate(EAP_only_ind = as.integer(term %in% EAP_only),
           HPL_only_ind = as.integer(term %in% HPL_only),
           MWS_only_ind = as.integer(term %in% MWS_only)) %>%
    group_by(ID) %>%
    summarise(EAP_only_count = sum(EAP_only_ind),
              HPL_only_count = sum(HPL_only_ind),
              MWS_only_count = sum(MWS_only_ind))

# Author pair only
EAP_HPL_only <- 
    unique(c(author_unigrams_tfidf %>% filter(author == "EAP" | author == "HPL", idf == log(1.5)) %>%
                 count(term, wt = n) %>% filter(n > 50) %>% .$term,
             author_bigrams_tfidf %>% filter(author == "EAP" | author == "HPL", idf == log(1.5)) %>%   
                 count(term, wt = n) %>% filter(n > 50) %>% .$term,
             author_trigrams_tfidf %>% filter(author == "EAP" | author == "HPL", idf == log(1.5)) %>%   
                 count(term, wt = n) %>% filter(n > 50) %>% .$term,
             author_tetragrams_tfidf %>% filter(author == "EAP" | author == "HPL", idf == log(1.5)) %>%   
                 count(term, wt = n) %>% filter(n > 50) %>% .$term,
             author_char_bigrams_tfidf %>% filter(author == "EAP" | author == "HPL", idf == log(1.5)) %>%
                 count(shingle, wt = n) %>% filter(n > 50) %>% .$shingle,
             author_char_trigrams_tfidf %>% filter(author == "EAP" | author == "HPL", idf == log(1.5)) %>%
                 count(shingle, wt = n) %>% filter(n > 50) %>% .$shingle,
             author_char_tetragrams_tfidf %>% filter(author == "EAP" | author == "HPL", idf == log(1.5)) %>%
                 count(shingle, wt = n) %>% filter(n > 50) %>% .$shingle,
             author_char_pentagrams_tfidf %>% filter(author == "EAP" | author == "HPL", idf == log(1.5)) %>%
                 count(shingle, wt = n) %>% filter(n > 50) %>% .$shingle))

EAP_MWS_only <- 
    unique(c(author_unigrams_tfidf %>% filter(author == "EAP" | author == "MWS", idf == log(1.5)) %>%
                 count(term, wt = n) %>% filter(n > 50) %>% .$term,
             author_bigrams_tfidf %>% filter(author == "EAP" | author == "MWS", idf == log(1.5)) %>%   
                 count(term, wt = n) %>% filter(n > 50) %>% .$term,
             author_trigrams_tfidf %>% filter(author == "EAP" | author == "MWS", idf == log(1.5)) %>%   
                 count(term, wt = n) %>% filter(n > 50) %>% .$term,
             author_tetragrams_tfidf %>% filter(author == "EAP" | author == "MWS", idf == log(1.5)) %>%   
                 count(term, wt = n) %>% filter(n > 50) %>% .$term,
             author_char_bigrams_tfidf %>% filter(author == "EAP" | author == "MWS", idf == log(1.5)) %>%
                 count(shingle, wt = n) %>% filter(n > 50) %>% .$shingle,
             author_char_trigrams_tfidf %>% filter(author == "EAP" | author == "MWS", idf == log(1.5)) %>%
                 count(shingle, wt = n) %>% filter(n > 50) %>% .$shingle,
             author_char_tetragrams_tfidf %>% filter(author == "EAP" | author == "MWS", idf == log(1.5)) %>%
                 count(shingle, wt = n) %>% filter(n > 50) %>% .$shingle,
             author_char_pentagrams_tfidf %>% filter(author == "EAP" | author == "MWS", idf == log(1.5)) %>%
                 count(shingle, wt = n) %>% filter(n > 50) %>% .$shingle))

HPL_MWS_only <- 
    unique(c(author_unigrams_tfidf %>% filter(author == "HPL" | author == "MWS", idf == log(1.5)) %>%
                 count(term, wt = n) %>% filter(n > 50) %>% .$term,
             author_bigrams_tfidf %>% filter(author == "HPL" | author == "MWS", idf == log(1.5)) %>%   
                 count(term, wt = n) %>% filter(n > 50) %>% .$term,
             author_trigrams_tfidf %>% filter(author == "HPL" | author == "MWS", idf == log(1.5)) %>%   
                 count(term, wt = n) %>% filter(n > 50) %>% .$term,
             author_tetragrams_tfidf %>% filter(author == "HPL" | author == "MWS", idf == log(1.5)) %>%   
                 count(term, wt = n) %>% filter(n > 50) %>% .$term,
             author_char_bigrams_tfidf %>% filter(author == "HPL" | author == "MWS", idf == log(1.5)) %>%
                 count(shingle, wt = n) %>% filter(n > 50) %>% .$shingle,
             author_char_trigrams_tfidf %>% filter(author == "HPL" | author == "MWS", idf == log(1.5)) %>%
                 count(shingle, wt = n) %>% filter(n > 50) %>% .$shingle,
             author_char_tetragrams_tfidf %>% filter(author == "HPL" | author == "MWS", idf == log(1.5)) %>%
                 count(shingle, wt = n) %>% filter(n > 50) %>% .$shingle,
             author_char_pentagrams_tfidf %>% filter(author == "HPL" | author == "MWS", idf == log(1.5)) %>%
                 count(shingle, wt = n) %>% filter(n > 50) %>% .$shingle))

train_author_pair_only <- map_df(1:4, ~ unnest_tokens(train, term, text, token = "ngrams", n = .x)) %>%
    bind_rows(map_df(2:5, ~ unnest_tokens(train, term, text, token = "character_shingles", n = .x, 
                                          strip_non_alphanum = FALSE))) %>%
    mutate(EAP_HPL_only_ind = as.integer(term %in% EAP_HPL_only),
           EAP_MWS_only_ind = as.integer(term %in% EAP_MWS_only),
           HPL_MWS_only_ind = as.integer(term %in% HPL_MWS_only)) %>%
    group_by(ID) %>%
    summarise(EAP_HPL_only_count = sum(EAP_HPL_only_ind),
              EAP_MWS_only_count = sum(EAP_MWS_only_ind),
              HPL_MWS_only_count = sum(HPL_MWS_only_ind))

test_author_pair_only <- map_df(1:4, ~ unnest_tokens(test, term, text, token = "ngrams", n = .x)) %>%
    bind_rows(map_df(2:5, ~ unnest_tokens(test, term, text, token = "character_shingles", n = .x, 
                                          strip_non_alphanum = FALSE))) %>%
    mutate(EAP_HPL_only_ind = as.integer(term %in% EAP_HPL_only),
           EAP_MWS_only_ind = as.integer(term %in% EAP_MWS_only),
           HPL_MWS_only_ind = as.integer(term %in% HPL_MWS_only)) %>%
    group_by(ID) %>%
    summarise(EAP_HPL_only_count = sum(EAP_HPL_only_ind),
              EAP_MWS_only_count = sum(EAP_MWS_only_ind),
              HPL_MWS_only_count = sum(HPL_MWS_only_ind))

ausp_plot <- train %>% 
    left_join(train_author_only, by = "ID") %>%
    left_join(train_author_pair_only, by = "ID") %>%
    select(-text,-ID) %>% 
    group_by(author) %>%
    summarise_all(mean) %>%
    gather(feature, value, -author) %>%
    ggplot(aes(x = feature, y = value, color = author, fill = author, size = value)) +
    geom_point(alpha = 0.6) +
    labs(x = "Feature", y = "Mean Value by Author") +
    coord_flip()
ggplotly(ausp_plot, tooltip = c("x","y","fill"))

# Sentiment features
train_senti <- train %>% unnest_tokens(term, text) %>%
    inner_join(get_sentiments("nrc"), by = c("term" = "word")) %>%
    count(ID, sentiment) %>%
    spread(sentiment, n, sep = "_", fill = 0)

test_senti <- test %>% unnest_tokens(term, text) %>%
    inner_join(get_sentiments("nrc"), by = c("term" = "word")) %>%
    count(ID, sentiment) %>%
    spread(sentiment, n, sep = "_", fill = 0)

# AFINN sentiment
afinn_sentiment <- train %>% select(-author) %>% bind_rows(test) %>%
    unnest_tokens(term, text) %>%
    inner_join(get_sentiments("afinn"), by = c("term" = "word")) %>%
    group_by(ID) %>%
    summarise(sentiment_afinn = mean(value))

train_senti <- train_senti %>% left_join(afinn_sentiment, by = "ID") %>%
    replace_na(list(sentiment_afinn = 0))

test_senti <- test_senti %>% left_join(afinn_sentiment, by = "ID") %>%
    replace_na(list(sentiment_afinn = 0))

rm(afinn_sentiment)

senti_plot <- train %>%
    right_join(train_senti, by = "ID") %>%
    select(-text,-ID) %>% 
    group_by(author) %>%
    summarise_all(mean) %>%
    gather(feature, value, -author) %>%
    ggplot(aes(x = feature, y = value, color = author, fill = author, size = value)) +
    geom_point(alpha = 0.6) +
    labs(x = "Feature", y = "Mean Value by Author") +
    coord_flip()
ggplotly(senti_plot, tooltip = c("x","y","fill"))

# Remove starting comma and period from train text
train$text <- str_replace(train$text, "^, ","")
train$text <- str_replace(train$text, "^\\.\" ","")

# Removing quotation marks from train and test text
train$text <- str_replace_all(train$text, "\"","")
test$text <- str_replace_all(test$text, "\"","")

# Penn Treebank POS annotation
tagger_pos <- rdr_model(language = "English", annotation = "POS")

tr_ch <- split(train, (as.numeric(rownames(train))-1) %/% 1000)
te_ch <- split(test, (as.numeric(rownames(test))-1) %/% 1000)

rst <- function(j, df){
    fx <- function(i) rdr_pos(tagger_pos, x = df$text[i], doc_id = df$ID[i])
    t <- mclapply(1:nrow(df), fx, mc.cores = 12)
}

train_pos_tag_list <- list()
for (j in 1:length(tr_ch)) {
    t <- rst(j, tr_ch[[j]])
    train_pos_tag_list <- c(train_pos_tag_list, t)
}
train_pos_tag <- bind_rows(train_pos_tag_list)

test_pos_tag_list <- list()
for (j in 1:length(te_ch)) {
    t <- rst(j, te_ch[[j]])
    test_pos_tag_list <- c(test_pos_tag_list, t)
}
test_pos_tag <- bind_rows(test_pos_tag_list)

train_pos_tag <- rdr_pos(tagger_pos, x = train$text, doc_id = train$ID)
test_pos_tag <- rdr_pos(tagger_pos, x = test$text, doc_id = test$ID)

# Universal POS annotation
tagger_upos <- rdr_model(language = "English", annotation = "UniversalPOS")

rst <- function(j, df){
    fx <- function(i) rdr_pos(tagger_upos, x = df$text[i], doc_id = df$ID[i])
    t <- mclapply(1:nrow(df), fx, mc.cores = 12)
}

train_upos_tag_list <- list()
for (j in 1:length(tr_ch)) {
    t <- rst(j, tr_ch[[j]])
    train_upos_tag_list <- c(train_upos_tag_list, t)
}
train_upos_tag <- bind_rows(train_upos_tag_list)

test_upos_tag_list <- list()
for (j in 1:length(te_ch)) {
    t <- rst(j, te_ch[[j]])
    test_upos_tag_list <- c(test_upos_tag_list, t)
}
test_upos_tag <- bind_rows(test_upos_tag_list)

train_upos_tag <- rdr_pos(tagger_upos, x = train$text, doc_id = train$ID)
test_upos_tag <- rdr_pos(tagger_upos, x = test$text, doc_id = test$ID)

# Penn Treebank POS tag count for each ID
train_pos_count <- train_pos_tag %>%
    count(doc_id, pos) %>%
    spread(pos, n, fill = 0) %>%
    rename(ID = doc_id)

test_pos_count <- test_pos_tag %>%
    count(doc_id, pos) %>%
    spread(pos, n, fill = 0) %>% 
    select(-c(`'`, `,`, `.`, `:`)) %>%
    rename(ID = doc_id)

train_pos_count <- train_pos_count %>% 
    select(intersect(colnames(.), colnames(test_pos_count)))

# Universal POS tag count for each ID
train_upos_count <- train_upos_tag %>%
    count(doc_id, pos) %>%
    spread(pos, n, fill = 0) %>% 
    select(-PUNCT) %>%
    rename(ID = doc_id)

test_upos_count <- test_upos_tag %>%
    count(doc_id, pos) %>%
    spread(pos, n, fill = 0) %>% 
    select(-PUNCT) %>%
    rename(ID = doc_id)

pos_plot <- train %>%  
    left_join(train_pos_count, by = "ID") %>%
    select(-text,-ID) %>%
    group_by(author) %>%
    summarise_all(mean) %>%
    gather(feature, value, -author) %>%
    ggplot(aes(x = feature, y = value, color = author, fill = author, size = value)) +
    geom_point(alpha = 0.6) +
    labs(x = "Feature", y = "Mean Value by Author") +
    coord_flip()
ggplotly(pos_plot, tooltip = c("x","y","fill"))

upos_plot <- train %>% 
    left_join(train_upos_count, by = "ID") %>%
    select(-text,-ID) %>%
    group_by(author) %>%
    summarise_all(mean) %>%
    gather(feature, value, -author) %>%
    ggplot(aes(x = feature, y = value, color = author, fill = author, size = value)) +
    geom_point(alpha = 0.6) +
    labs(x = "Feature", y = "Mean Value by Author") +
    coord_flip()
ggplotly(upos_plot, tooltip = c("x","y","fill"))

# Create folds for stacking
set.seed(2468)
folds <- createFolds(train$author, k = 5)

# One hot encoding target
y_train <- to_categorical(as.integer(as.factor(train$author)))
y_train <- y_train[,2:4]

# word unigrams in train
word1G <- train %>% select(-author) %>% 
    unnest_tokens(token, text, token = "words", to_lower = FALSE, strip_punct = FALSE) %>% 
    select(-ID) %>%
    count(token) %>%
    .$token

# Document term matrix of word unigrams for train and test combined
word1Gcount <- train %>% select(-author) %>% bind_rows(test) %>%
    unnest_tokens(token, text, token = "words", to_lower = FALSE, strip_punct = FALSE) %>%
    count(ID, token) %>%
    filter(token %in% word1G) %>%
    cast_dtm(ID, token, n)

# Separate train and test data
x_train_word1Gcount <- word1Gcount[train$ID,] %>% as.matrix
x_test_word1Gcount <- word1Gcount[test$ID,] %>% as.matrix
rm(word1Gcount,word1G)

# Function to build model, make predictions and evaluate on given fold and test data
word1GcountModel <- function(x_train_word1Gcount, x_test_word1Gcount, fold, y_train){
    sai_word1Gcount <- keras_model_sequential() %>%
        layer_dense(units = 16, activation = "relu", input_shape = ncol(x_train_word1Gcount)) %>% 
        layer_dense(units = 16, activation = "relu") %>%
        layer_dense(units = 3, activation = 'softmax')
    
    sai_word1Gcount %>% compile(
        loss = 'categorical_crossentropy',
        optimizer = 'rmsprop',
        metrics = c('accuracy')
    )
    
    history_sai_word1Gcount <- sai_word1Gcount %>% fit(
        x_train_word1Gcount[-fold,], y_train[-fold,],
        batch_size = 2^9,
        epochs = 20,
        validation_split = 0.1, 
        verbose = FALSE,
        callbacks = list(
            callback_early_stopping(monitor = "val_loss", patience = 2),
            callback_model_checkpoint(
                filepath = "word1Gcount.hdf5",
                monitor = "val_loss",
                mode = "min",
                save_best_only = TRUE)
        )
    )
    sai_word1Gcount <- load_model_hdf5("word1Gcount.hdf5")
    
    train_pred <- sai_word1Gcount %>% predict(x_train_word1Gcount[fold,])
    test_pred <- sai_word1Gcount %>% predict(x_test_word1Gcount)
    fold_eval <- sai_word1Gcount %>% evaluate(x_train_word1Gcount[fold,], y_train[fold,])
    out <- list(train_pred = train_pred, test_pred = test_pred, logloss = fold_eval$loss, acc = fold_eval$acc )
    k_clear_session()
    return(out)
}

train_word1Gcount <- matrix(0, nrow = nrow(train), ncol = 3)
test_word1Gcount <- matrix(0, nrow = nrow(test), ncol = 3)
metrics_word1Gcount <- matrix(0, 5, 2)

# Form dataframes of stacked predictions, average predictions on test and evaluation metrics
for(i in 1:5){
    results_word1Gcount <- word1GcountModel(x_train_word1Gcount, x_test_word1Gcount, folds[[i]], y_train)
    train_word1Gcount[folds[[i]], ] <- results_word1Gcount$train_pred
    test_word1Gcount <- test_word1Gcount + (results_word1Gcount$test_pred)/5
    metrics_word1Gcount[i,1] <- results_word1Gcount$logloss
    metrics_word1Gcount[i,2] <- results_word1Gcount$acc
} 

train_word1Gcount <- train_word1Gcount %>% as.data.frame() %>%
    rename(word1Gcount_EAP=V1, word1Gcount_HPL=V2, word1Gcount_MWS=V3)
test_word1Gcount <- test_word1Gcount %>% as.data.frame() %>%
    rename(word1Gcount_EAP=V1, word1Gcount_HPL=V2, word1Gcount_MWS=V3)
metrics_word1Gcount <- metrics_word1Gcount %>% as.data.frame() %>% 
    rename(logloss=V1, acc=V2) 
rownames(metrics_word1Gcount) <- paste0("fold ", 1:5, ":")
metrics_word1Gcount

# word bigrams in train
word2G <- train %>% select(-author) %>% 
    unnest_tokens(token, text, token = "ngrams", n=2, to_lower = FALSE) %>% select(-ID) %>%
    count(token) %>%
    filter(n > 2) %>%
    .$token

# Document term matrix of word bigrams for train and test combined
word2Gcount <- train %>% select(-author) %>% bind_rows(test) %>%
    unnest_tokens(token, text, token = "ngrams", n=2, to_lower = FALSE) %>%
    count(ID, token) %>%
    filter(token %in% word2G) %>%
    cast_dtm(ID, token, n) %>%
    as.matrix()

# Append rows with zero columns in word2Gcount for excerpts that don't contain word2G
rowsRemoved <- setdiff(c(train$ID,test$ID),rownames(word2Gcount))
allZeros <- matrix(0, length(rowsRemoved), ncol(word2Gcount), 
                   dimnames = list(rowsRemoved, colnames(word2Gcount)))
word2Gcount <- word2Gcount %>% rbind(allZeros)
rm(rowsRemoved,allZeros)

# Separate train and test data
x_train_word2Gcount <- word2Gcount[train$ID,] %>% as.matrix
x_test_word2Gcount <- word2Gcount[test$ID,] %>% as.matrix
rm(word2Gcount,word2G)

# Function to build model, make predictions and evaluate on given fold and test data
word2GcountModel <- function(x_train_word2Gcount, x_test_word2Gcount, fold, y_train){
    sai_word2Gcount <- keras_model_sequential() %>%
        layer_dense(units = 16, activation = "relu", input_shape = ncol(x_train_word2Gcount)) %>%
        layer_dense(units = 16, activation = "relu") %>%
        layer_dense(units = 3, activation = 'softmax')
    
    sai_word2Gcount %>% compile(
        loss = 'categorical_crossentropy',
        optimizer = 'rmsprop',
        metrics = c('accuracy')
    )
    
    history_sai_word2Gcount <- sai_word2Gcount %>% fit(
        x_train_word2Gcount[-fold,], y_train[-fold,],
        batch_size = 2^8,
        epochs = 20,
        validation_split = 0.1,
        verbose = FALSE,
        callbacks = list(
            callback_early_stopping(monitor = "val_loss", patience = 2),
            callback_model_checkpoint(
                filepath = "word2Gcount.hdf5",
                monitor = "val_loss",
                mode = "min",
                save_best_only = TRUE)
        )
    )
    sai_word2Gcount <- load_model_hdf5("word2Gcount.hdf5")
    
    train_pred <- sai_word2Gcount %>% predict(x_train_word2Gcount[fold,])
    test_pred <- sai_word2Gcount %>% predict(x_test_word2Gcount)
    fold_eval <- sai_word2Gcount %>% evaluate(x_train_word2Gcount[fold,], y_train[fold,])
    out <- list(train_pred = train_pred, test_pred = test_pred, logloss = fold_eval$loss, acc = fold_eval$acc )
    k_clear_session()
    return(out)
}

train_word2Gcount <- matrix(0, nrow = nrow(train), ncol = 3)
test_word2Gcount <- matrix(0, nrow = nrow(test), ncol = 3)
metrics_word2Gcount <- matrix(0, 5, 2)

# Form dataframes of stacked predictions, average predictions on test and evaluation metrics
for(i in 1:5){
    results_word2Gcount <- word2GcountModel(x_train_word2Gcount, x_test_word2Gcount, folds[[i]], y_train)
    train_word2Gcount[folds[[i]], ] <- results_word2Gcount$train_pred
    test_word2Gcount <- test_word2Gcount + (results_word2Gcount$test_pred)/5
    metrics_word2Gcount[i,1] <- results_word2Gcount$logloss
    metrics_word2Gcount[i,2] <- results_word2Gcount$acc
}

train_word2Gcount <- train_word2Gcount %>% as.data.frame() %>%
    rename(word2Gcount_EAP=V1, word2Gcount_HPL=V2, word2Gcount_MWS=V3)
test_word2Gcount <- test_word2Gcount %>% as.data.frame() %>%
    rename(word2Gcount_EAP=V1, word2Gcount_HPL=V2, word2Gcount_MWS=V3)
metrics_word2Gcount <- metrics_word2Gcount %>% as.data.frame() %>% rename(logloss=V1, acc=V2)
rownames(metrics_word2Gcount) <- paste0("fold ", 1:5, ":")
metrics_word2Gcount

# word trigrams in train
word3G <- train %>% select(-author) %>% 
    unnest_tokens(token, text, token = "ngrams", n=3, to_lower = FALSE) %>% select(-ID) %>%
    count(token) %>%
    filter(n > 2) %>%
    .$token

# Document term matrix of word trigrams for train and test combined
word3Gcount <- train %>% select(-author) %>% bind_rows(test) %>%
    unnest_tokens(token, text, token = "ngrams", n=3, to_lower = FALSE) %>%
    count(ID, token) %>%
    filter(token %in% word3G) %>%
    cast_dtm(ID, token, n) %>%
    as.matrix()

# Append rows with zero columns in word3Gcount for excerpts that don't contain word3G
rowsRemoved <- setdiff(c(train$ID,test$ID),rownames(word3Gcount))
allZeros <- matrix(0, length(rowsRemoved), ncol(word3Gcount), 
                   dimnames = list(rowsRemoved, colnames(word3Gcount)))
word3Gcount <- word3Gcount %>% rbind(allZeros)
rm(rowsRemoved,allZeros)

# Separate train and test data
x_train_word3Gcount <- word3Gcount[train$ID,] %>% as.matrix
x_test_word3Gcount <- word3Gcount[test$ID,] %>% as.matrix
rm(word3Gcount,word3G)

# Function to build model, make predictions and evaluate on given fold and test data
word3GcountModel <- function(x_train_word3Gcount, x_test_word3Gcount, fold, y_train){
    sai_word3Gcount <- keras_model_sequential() %>%
        layer_dense(units = 16, activation = "relu", input_shape = ncol(x_train_word3Gcount)) %>%
        layer_dense(units = 16, activation = "relu") %>%
        layer_dense(units = 3, activation = 'softmax')
    
    sai_word3Gcount %>% compile(
        loss = 'categorical_crossentropy',
        optimizer = 'rmsprop',
        metrics = c('accuracy')
    )
    
    history_sai_word3Gcount <- sai_word3Gcount %>% fit(
        x_train_word3Gcount[-fold,], y_train[-fold,],
        batch_size = 2^9,
        epochs = 20,
        validation_split = 0.1,
        verbose = FALSE,
        callbacks = list(
            callback_early_stopping(monitor = "val_loss", patience = 2),
            callback_model_checkpoint(
                filepath = "word3Gcount.hdf5",
                monitor = "val_loss",
                mode = "min",
                save_best_only = TRUE)
        )
    )
    sai_word3Gcount <- load_model_hdf5("word3Gcount.hdf5")
    
    train_pred <- sai_word3Gcount %>% predict(x_train_word3Gcount[fold,])
    test_pred <- sai_word3Gcount %>% predict(x_test_word3Gcount)
    fold_eval <- sai_word3Gcount %>% evaluate(x_train_word3Gcount[fold,], y_train[fold,])
    out <- list(train_pred = train_pred, test_pred = test_pred, logloss = fold_eval$loss, acc = fold_eval$acc )
    k_clear_session()
    return(out)
}

train_word3Gcount <- matrix(0, nrow = nrow(train), ncol = 3)
test_word3Gcount <- matrix(0, nrow = nrow(test), ncol = 3)
metrics_word3Gcount <- matrix(0, 5, 2)

# Form dataframes of stacked predictions, average predictions on test and evaluation metrics
for(i in 1:5){
    results_word3Gcount <- word3GcountModel(x_train_word3Gcount, x_test_word3Gcount, folds[[i]], y_train)
    train_word3Gcount[folds[[i]], ] <- results_word3Gcount$train_pred
    test_word3Gcount <- test_word3Gcount + (results_word3Gcount$test_pred)/5
    metrics_word3Gcount[i,1] <- results_word3Gcount$logloss
    metrics_word3Gcount[i,2] <- results_word3Gcount$acc
}

train_word3Gcount <- train_word3Gcount %>% as.data.frame() %>%
    rename(word3Gcount_EAP=V1, word3Gcount_HPL=V2, word3Gcount_MWS=V3)
test_word3Gcount <- test_word3Gcount %>% as.data.frame() %>%
    rename(word3Gcount_EAP=V1, word3Gcount_HPL=V2, word3Gcount_MWS=V3)
metrics_word3Gcount <- metrics_word3Gcount %>% as.data.frame() %>% rename(logloss=V1, acc=V2)
rownames(metrics_word3Gcount) <- paste0("fold ", 1:5, ":")
metrics_word3Gcount
# word tetragrams in train
word4G <- train %>% select(-author) %>% 
    unnest_tokens(token, text, token = "ngrams", n=3, to_lower = FALSE) %>% select(-ID) %>%
    count(token) %>%
    filter(n > 2) %>%
    .$token

# Document term matrix of word tetragrams for train and test combined
word4Gcount <- train %>% select(-author) %>% bind_rows(test) %>%
    unnest_tokens(token, text, token = "ngrams", n=3, to_lower = FALSE) %>%
    count(ID, token) %>%
    filter(token %in% word4G) %>%
    cast_dtm(ID, token, n) %>%
    as.matrix()

# Append rows with zero columns in word4Gcount for excerpts that don't contain word4G
rowsRemoved <- setdiff(c(train$ID,test$ID),rownames(word4Gcount))
allZeros <- matrix(0, length(rowsRemoved), ncol(word4Gcount), 
                   dimnames = list(rowsRemoved, colnames(word4Gcount)))
word4Gcount <- word4Gcount %>% rbind(allZeros)
rm(rowsRemoved,allZeros)

# Separate train and test data
x_train_word4Gcount <- word4Gcount[train$ID,] %>% as.matrix
x_test_word4Gcount <- word4Gcount[test$ID,] %>% as.matrix
rm(word4Gcount,word4G)

# Function to build model, make predictions and evaluate on given fold and test data
word4GcountModel <- function(x_train_word4Gcount, x_test_word4Gcount, fold, y_train){
    sai_word4Gcount <- keras_model_sequential() %>%
        layer_dense(units = 18, activation = "relu", input_shape = ncol(x_train_word4Gcount)) %>%
        layer_dense(units = 18, activation = "relu") %>%
        layer_dense(units = 3, activation = 'softmax')
    
    sai_word4Gcount %>% compile(
        loss = 'categorical_crossentropy',
        optimizer = 'rmsprop',
        metrics = c('accuracy')
    )
    
    history_sai_word4Gcount <- sai_word4Gcount %>% fit(
        x_train_word4Gcount[-fold,], y_train[-fold,],
        batch_size = 2^9,
        epochs = 20,
        validation_split = 0.1,
        verbose = FALSE,
        callbacks = list(
            callback_early_stopping(monitor = "val_loss", patience = 2),
            callback_model_checkpoint(
                filepath = "word4Gcount.hdf5",
                monitor = "val_loss",
                mode = "min",
                save_best_only = TRUE)
        )
    )
    sai_word4Gcount <- load_model_hdf5("word4Gcount.hdf5")
    
    train_pred <- sai_word4Gcount %>% predict(x_train_word4Gcount[fold,])
    test_pred <- sai_word4Gcount %>% predict(x_test_word4Gcount)
    fold_eval <- sai_word4Gcount %>% evaluate(x_train_word4Gcount[fold,], y_train[fold,])
    out <- list(train_pred = train_pred, test_pred = test_pred, logloss = fold_eval$loss, acc = fold_eval$acc )
    k_clear_session()
    return(out)
}

train_word4Gcount <- matrix(0, nrow = nrow(train), ncol = 3)
test_word4Gcount <- matrix(0, nrow = nrow(test), ncol = 3)
metrics_word4Gcount <- matrix(0, 5, 2)

# Form dataframes of stacked predictions, average predictions on test and evaluation metrics
for(i in 1:5){
    results_word4Gcount <- word4GcountModel(x_train_word4Gcount, x_test_word4Gcount, folds[[i]], y_train)
    train_word4Gcount[folds[[i]], ] <- results_word4Gcount$train_pred
    test_word4Gcount <- test_word4Gcount + (results_word4Gcount$test_pred)/5
    metrics_word4Gcount[i,1] <- results_word4Gcount$logloss
    metrics_word4Gcount[i,2] <- results_word4Gcount$acc
} 

train_word4Gcount <- train_word4Gcount %>% as.data.frame() %>%
    rename(word4Gcount_EAP=V1, word4Gcount_HPL=V2, word4Gcount_MWS=V3)
test_word4Gcount <- test_word4Gcount %>% as.data.frame() %>%
    rename(word4Gcount_EAP=V1, word4Gcount_HPL=V2, word4Gcount_MWS=V3)
metrics_word4Gcount <- metrics_word4Gcount %>% as.data.frame() %>% rename(logloss=V1, acc=V2)
rownames(metrics_word4Gcount) <- paste0("fold ", 1:5, ":")
metrics_word4Gcount

# char 123grams in train
char123G <- map_df(1:3, ~ unnest_tokens(train %>% select(-author), token, text, 
                                        token = "character_shingles", n = .x, to_lower = FALSE, 
                                        lowercase = FALSE, strip_non_alphanum = FALSE)) %>%
    select(-ID) %>%
    count(token) %>%
    .$token

# Document term matrix of char 123grams for train and test combined
char123Gcount <- map_df(1:3, ~ unnest_tokens(train %>% select(-author) %>% bind_rows(test), token, text, 
                                             token = "character_shingles", n = .x, to_lower = FALSE, 
                                             lowercase = FALSE, strip_non_alphanum = FALSE)) %>%
    count(ID, token) %>%
    filter(token %in% char123G) %>%
    cast_dtm(ID, token, n) %>%
    as.matrix()

# Separate train and test data
x_train_char123Gcount <- char123Gcount[train$ID,] %>% as.matrix
x_test_char123Gcount <- char123Gcount[test$ID,] %>% as.matrix
rm(char123Gcount,char123G)

# Function to build model, make predictions and evaluate on given fold and test data
char123GcountModel <- function(x_train_char123Gcount, x_test_char123Gcount, fold, y_train){
    sai_char123Gcount <- keras_model_sequential() %>%
        layer_dense(units = 16, activation = "relu", input_shape = ncol(x_train_char123Gcount)) %>%
        layer_dense(units = 16, activation = "relu") %>%
        layer_dense(units = 3, activation = 'softmax')
    
    sai_char123Gcount %>% compile(
        loss = 'categorical_crossentropy',
        optimizer = 'rmsprop',
        metrics = c('accuracy')
    )
    
    history_sai_char123Gcount <- sai_char123Gcount %>% fit(
        x_train_char123Gcount[-fold,], y_train[-fold,],
        batch_size = 2^8,
        epochs = 20,
        validation_split = 0.1,
        verbose = FALSE,
        callbacks = list(
            callback_early_stopping(monitor = "val_loss", patience = 2),
            callback_model_checkpoint(
                filepath = "char123Gcount.hdf5",
                monitor = "val_loss",
                mode = "min",
                save_best_only = TRUE)
        )
    )
    sai_char123Gcount <- load_model_hdf5("char123Gcount.hdf5")
    
    train_pred <- sai_char123Gcount %>% predict(x_train_char123Gcount[fold,])
    test_pred <- sai_char123Gcount %>% predict(x_test_char123Gcount)
    fold_eval <- sai_char123Gcount %>% evaluate(x_train_char123Gcount[fold,], y_train[fold,])
    out <- list(train_pred = train_pred, test_pred = test_pred, logloss = fold_eval$loss, acc = fold_eval$acc )
    k_clear_session()
    return(out)
}

train_char123Gcount <- matrix(0, nrow = nrow(train), ncol = 3)
test_char123Gcount <- matrix(0, nrow = nrow(test), ncol = 3)
metrics_char123Gcount <- matrix(0, 5, 2)

# Form dataframes of stacked predictions, average predictions on test and evaluation metrics
for(i in 1:5){
    results_char123Gcount <- char123GcountModel(x_train_char123Gcount, x_test_char123Gcount, 
                                                folds[[i]], y_train)
    train_char123Gcount[folds[[i]], ] <- results_char123Gcount$train_pred
    test_char123Gcount <- test_char123Gcount + (results_char123Gcount$test_pred)/5
    metrics_char123Gcount[i,1] <- results_char123Gcount$logloss
    metrics_char123Gcount[i,2] <- results_char123Gcount$acc
}

train_char123Gcount <- train_char123Gcount %>% as.data.frame() %>%
    rename(char123Gcount_EAP=V1, char123Gcount_HPL=V2, char123Gcount_MWS=V3)
test_char123Gcount <- test_char123Gcount %>% as.data.frame() %>%
    rename(char123Gcount_EAP=V1, char123Gcount_HPL=V2, char123Gcount_MWS=V3)
metrics_char123Gcount <- metrics_char123Gcount %>% as.data.frame() %>% rename(logloss=V1, acc=V2)
rownames(metrics_char123Gcount) <- paste0("fold ", 1:5, ":")
metrics_char123Gcount

# char tetragrams in train
char4G <- train %>% select(-author) %>% 
    unnest_tokens(token, text, token = "character_shingles", n=4, to_lower = FALSE, 
                  lowercase = FALSE, strip_non_alphanum = FALSE) %>%
    select(-ID) %>%
    count(token) %>%
    filter(n > 3) %>%
    .$token

# Document term matrix of char tetragrams for train and test combined
char4Gcount <- train %>% select(-author) %>% bind_rows(test) %>%
    unnest_tokens(token, text, token = "character_shingles", n=4, to_lower = FALSE, 
                  lowercase = FALSE, strip_non_alphanum = FALSE) %>%
    count(ID, token) %>%
    filter(token %in% char4G) %>%
    cast_dtm(ID, token, n) %>%
    as.matrix()

# Append rows with zero columns in char4Gcount for excerpts that don't contain char4G
rowsRemoved <- setdiff(c(train$ID,test$ID),rownames(char4Gcount))
allZeros <- matrix(0, length(rowsRemoved), ncol(char4Gcount), 
                   dimnames = list(rowsRemoved, colnames(char4Gcount)))
char4Gcount <- char4Gcount %>% rbind(allZeros)
rm(rowsRemoved,allZeros)

# Separate train and test data
x_train_char4Gcount <- char4Gcount[train$ID,] %>% as.matrix
x_test_char4Gcount <- char4Gcount[test$ID,] %>% as.matrix
rm(char4Gcount,char4G)

# Function to build model, make predictions and evaluate on given fold and test data
char4GcountModel <- function(x_train_char4Gcount, x_test_char4Gcount, fold, y_train){
    sai_char4Gcount <- keras_model_sequential() %>%
        layer_dense(units = 16, activation = "relu", input_shape = ncol(x_train_char4Gcount)) %>%
        layer_dense(units = 16, activation = "relu") %>%
        layer_dense(units = 3, activation = 'softmax')
    
    sai_char4Gcount %>% compile(
        loss = 'categorical_crossentropy',
        optimizer = 'rmsprop',
        metrics = c('accuracy')
    )
    
    history_sai_char4Gcount <- sai_char4Gcount %>% fit(
        x_train_char4Gcount[-fold,], y_train[-fold,],
        batch_size = 2^8,
        epochs = 20,
        validation_split = 0.1,
        verbose = FALSE,
        callbacks = list(
            callback_early_stopping(monitor = "val_loss", patience = 2),
            callback_model_checkpoint(
                filepath = "char4Gcount.hdf5",
                monitor = "val_loss",
                mode = "min",
                save_best_only = TRUE)
        )
    )
    sai_char4Gcount <- load_model_hdf5("char4Gcount.hdf5")
    
    train_pred <- sai_char4Gcount %>% predict(x_train_char4Gcount[fold,])
    test_pred <- sai_char4Gcount %>% predict(x_test_char4Gcount)
    fold_eval <- sai_char4Gcount %>% evaluate(x_train_char4Gcount[fold,], y_train[fold,])
    out <- list(train_pred = train_pred, test_pred = test_pred, logloss = fold_eval$loss, acc = fold_eval$acc )
    k_clear_session()
    return(out)
}

train_char4Gcount <- matrix(0, nrow = nrow(train), ncol = 3)
test_char4Gcount <- matrix(0, nrow = nrow(test), ncol = 3)
metrics_char4Gcount <- matrix(0, 5, 2)

# Form dataframes of stacked predictions, average predictions on test and evaluation metrics
for(i in 1:5){
    results_char4Gcount <- char4GcountModel(x_train_char4Gcount, x_test_char4Gcount, folds[[i]], y_train)
    train_char4Gcount[folds[[i]], ] <- results_char4Gcount$train_pred
    test_char4Gcount <- test_char4Gcount + (results_char4Gcount$test_pred)/5
    metrics_char4Gcount[i,1] <- results_char4Gcount$logloss
    metrics_char4Gcount[i,2] <- results_char4Gcount$acc
}

train_char4Gcount <- train_char4Gcount %>% as.data.frame() %>%
    rename(char4Gcount_EAP=V1, char4Gcount_HPL=V2, char4Gcount_MWS=V3)
test_char4Gcount <- test_char4Gcount %>% as.data.frame() %>%
    rename(char4Gcount_EAP=V1, char4Gcount_HPL=V2, char4Gcount_MWS=V3)
metrics_char4Gcount <- metrics_char4Gcount %>% as.data.frame() %>% rename(logloss=V1, acc=V2)
rownames(metrics_char4Gcount) <- paste0("fold ", 1:5, ":")
metrics_char4Gcount

# char pentagrams in train for n > 4 and n <= 12
char5Gp1 <- train %>% select(-author) %>% 
    unnest_tokens(token, text, token = "character_shingles", n=5, to_lower = FALSE, 
                  lowercase = FALSE, strip_non_alphanum = FALSE) %>%
    select(-ID) %>%
    count(token) %>%
    filter(n > 4 & n <= 12) %>%
    .$token

# Document term matrix of char pentagrams for train and test combined
char5Gp1count <- train %>% select(-author) %>% bind_rows(test) %>%
    unnest_tokens(token, text, token = "character_shingles", n=5, to_lower = FALSE, 
                  lowercase = FALSE, strip_non_alphanum = FALSE) %>%
    count(ID, token) %>%
    filter(token %in% char5Gp1) %>%
    cast_dtm(ID, token, n) %>%
    as.matrix()

# Append rows with zero columns in char5Gp1count for excerpts that don't contain char5Gp1
rowsRemoved <- setdiff(c(train$ID,test$ID),rownames(char5Gp1count))
allZeros <- matrix(0, length(rowsRemoved), ncol(char5Gp1count), 
                   dimnames = list(rowsRemoved, colnames(char5Gp1count)))
char5Gp1count <- char5Gp1count %>% rbind(allZeros)
rm(rowsRemoved,allZeros)

# Separate train and test data
x_train_char5Gp1count <- char5Gp1count[train$ID,] %>% as.matrix
x_test_char5Gp1count <- char5Gp1count[test$ID,] %>% as.matrix
rm(char5Gp1count,char5Gp1)

# Function to build model, make predictions and evaluate on given fold and test data
char5Gp1countModel <- function(x_train_char5Gp1count, x_test_char5Gp1count, fold, y_train){
    sai_char5Gp1count <- keras_model_sequential() %>%
        layer_dense(units = 16, activation = "relu", input_shape = ncol(x_train_char5Gp1count)) %>%
        layer_dense(units = 16, activation = "relu") %>%
        layer_dense(units = 3, activation = 'softmax')
    
    sai_char5Gp1count %>% compile(
        loss = 'categorical_crossentropy',
        optimizer = 'rmsprop',
        metrics = c('accuracy')
    )
    
    history_sai_char5Gp1count <- sai_char5Gp1count %>% fit(
        x_train_char5Gp1count[-fold,], y_train[-fold,],
        batch_size = 2^9,
        epochs = 20,
        validation_split = 0.1,
        verbose = FALSE,
        callbacks = list(
            callback_early_stopping(monitor = "val_loss", patience = 2),
            callback_model_checkpoint(
                filepath = "char5Gp1count.hdf5",
                monitor = "val_loss",
                mode = "min",
                save_best_only = TRUE)
        )
    )
    sai_char5Gp1count <- load_model_hdf5("char5Gp1count.hdf5")
    
    train_pred <- sai_char5Gp1count %>% predict(x_train_char5Gp1count[fold,])
    test_pred <- sai_char5Gp1count %>% predict(x_test_char5Gp1count)
    fold_eval <- sai_char5Gp1count %>% evaluate(x_train_char5Gp1count[fold,], y_train[fold,])
    out <- list(train_pred = train_pred, test_pred = test_pred, logloss = fold_eval$loss, acc = fold_eval$acc )
    k_clear_session()
    return(out)
}

train_char5Gp1count <- matrix(0, nrow = nrow(train), ncol = 3)
test_char5Gp1count <- matrix(0, nrow = nrow(test), ncol = 3)
metrics_char5Gp1count <- matrix(0, 5, 2)

# Form dataframes of stacked predictions, average predictions on test and evaluation metrics
for(i in 1:5){
    results_char5Gp1count <- char5Gp1countModel(x_train_char5Gp1count, x_test_char5Gp1count, 
                                                folds[[i]], y_train)
    train_char5Gp1count[folds[[i]], ] <- results_char5Gp1count$train_pred
    test_char5Gp1count <- test_char5Gp1count + (results_char5Gp1count$test_pred)/5
    metrics_char5Gp1count[i,1] <- results_char5Gp1count$logloss
    metrics_char5Gp1count[i,2] <- results_char5Gp1count$acc
}

train_char5Gp1count <- train_char5Gp1count %>% as.data.frame() %>%
    rename(char5Gp1count_EAP=V1, char5Gp1count_HPL=V2, char5Gp1count_MWS=V3)
test_char5Gp1count <- test_char5Gp1count %>% as.data.frame() %>%
    rename(char5Gp1count_EAP=V1, char5Gp1count_HPL=V2, char5Gp1count_MWS=V3)
metrics_char5Gp1count <- metrics_char5Gp1count %>% as.data.frame() %>% rename(logloss=V1, acc=V2)
rownames(metrics_char5Gp1count) <- paste0("fold ", 1:5, ":")
metrics_char5Gp1count

# char pentagrams in train for n > 12
char5Gp2 <- train %>% select(-author) %>% 
    unnest_tokens(token, text, token = "character_shingles", n=5, to_lower = FALSE, 
                  lowercase = FALSE, strip_non_alphanum = FALSE) %>%
    select(-ID) %>%
    count(token) %>%
    filter(n > 12) %>%
    .$token

# Document term matrix of char pentagrams for train and test combined
char5Gp2count <- train %>% select(-author) %>% bind_rows(test) %>%
    unnest_tokens(token, text, token = "character_shingles", n=5, to_lower = FALSE, 
                  lowercase = FALSE, strip_non_alphanum = FALSE) %>%
    count(ID, token) %>%
    filter(token %in% char5Gp2) %>%
    cast_dtm(ID, token, n) %>%
    as.matrix()

# Append rows with zero columns in char5Gp2count for excerpts that don't contain char5Gp2
rowsRemoved <- setdiff(c(train$ID,test$ID),rownames(char5Gp2count))
allZeros <- matrix(0, length(rowsRemoved), ncol(char5Gp2count), 
                   dimnames = list(rowsRemoved, colnames(char5Gp2count)))
char5Gp2count <- char5Gp2count %>% rbind(allZeros)
rm(rowsRemoved,allZeros)

# Separate train and test data
x_train_char5Gp2count <- char5Gp2count[train$ID,] %>% as.matrix
x_test_char5Gp2count <- char5Gp2count[test$ID,] %>% as.matrix
rm(char5Gp2count,char5Gp2)

# Function to build model, make predictions and evaluate on given fold and test data
char5Gp2countModel <- function(x_train_char5Gp2count, x_test_char5Gp2count, fold, y_train){
    sai_char5Gp2count <- keras_model_sequential() %>%
        layer_dense(units = 16, activation = "relu", input_shape = ncol(x_train_char5Gp2count)) %>%
        layer_dense(units = 16, activation = "relu") %>%
        layer_dense(units = 3, activation = 'softmax')
    
    sai_char5Gp2count %>% compile(
        loss = 'categorical_crossentropy',
        optimizer = 'rmsprop',
        metrics = c('accuracy')
    )
    
    history_sai_char5Gp2count <- sai_char5Gp2count %>% fit(
        x_train_char5Gp2count[-fold,], y_train[-fold,],
        batch_size = 2^9,
        epochs = 20,
        validation_split = 0.1,
        verbose = FALSE,
        callbacks = list(
            callback_early_stopping(monitor = "val_loss", patience = 2),
            callback_model_checkpoint(
                filepath = "char5Gp2count.hdf5",
                monitor = "val_loss",
                mode = "min",
                save_best_only = TRUE)
        )
    )
    sai_char5Gp2count <- load_model_hdf5("char5Gp2count.hdf5")
    
    train_pred <- sai_char5Gp2count %>% predict(x_train_char5Gp2count[fold,])
    test_pred <- sai_char5Gp2count %>% predict(x_test_char5Gp2count)
    fold_eval <- sai_char5Gp2count %>% evaluate(x_train_char5Gp2count[fold,], y_train[fold,])
    out <- list(train_pred = train_pred, test_pred = test_pred, logloss = fold_eval$loss, acc = fold_eval$acc )
    k_clear_session()
    return(out)
}

train_char5Gp2count <- matrix(0, nrow = nrow(train), ncol = 3)
test_char5Gp2count <- matrix(0, nrow = nrow(test), ncol = 3)
metrics_char5Gp2count <- matrix(0, 5, 2)

# Form dataframes of stacked predictions, average predictions on test and evaluation metrics
for(i in 1:5){
    results_char5Gp2count <- char5Gp2countModel(x_train_char5Gp2count, x_test_char5Gp2count, 
                                                folds[[i]], y_train)
    train_char5Gp2count[folds[[i]], ] <- results_char5Gp2count$train_pred
    test_char5Gp2count <- test_char5Gp2count + (results_char5Gp2count$test_pred)/5
    metrics_char5Gp2count[i,1] <- results_char5Gp2count$logloss
    metrics_char5Gp2count[i,2] <- results_char5Gp2count$acc
}

train_char5Gp2count <- train_char5Gp2count %>% as.data.frame() %>%
    rename(char5Gp2count_EAP=V1, char5Gp2count_HPL=V2, char5Gp2count_MWS=V3)
test_char5Gp2count <- test_char5Gp2count %>% as.data.frame() %>%
    rename(char5Gp2count_EAP=V1, char5Gp2count_HPL=V2, char5Gp2count_MWS=V3)
metrics_char5Gp2count <- metrics_char5Gp2count %>% as.data.frame() %>% rename(logloss=V1, acc=V2)
rownames(metrics_char5Gp2count) <- paste0("fold ", 1:5, ":")
metrics_char5Gp2count

# char hexagrams in train for n > 6 & n <= 15
char6Gp1 <- train %>% select(-author) %>% 
    unnest_tokens(token, text, token = "character_shingles", n=6, to_lower = FALSE, 
                  lowercase = FALSE, strip_non_alphanum = FALSE) %>%
    select(-ID) %>%
    count(token) %>%
    filter(n > 7 & n <= 15) %>%
    .$token


# Document term matrix of char hexagrams for train and test combined for n > 6 & n <= 15
char6Gp1count <- train %>% select(-author) %>% bind_rows(test) %>%
    unnest_tokens(token, text, token = "character_shingles", n=6, to_lower = FALSE, 
                  lowercase = FALSE, strip_non_alphanum = FALSE) %>%
    count(ID, token) %>%
    filter(token %in% char6Gp1) %>%
    cast_dtm(ID, token, n) %>%
    as.matrix()

# Append rows with zero columns in char6Gp1count for excerpts that don't contain char6Gp1
rowsRemoved <- setdiff(c(train$ID,test$ID),rownames(char6Gp1count))
allZeros <- matrix(0, length(rowsRemoved), ncol(char6Gp1count), 
                   dimnames = list(rowsRemoved, colnames(char6Gp1count)))
char6Gp1count <- char6Gp1count %>% rbind(allZeros)
rm(rowsRemoved,allZeros)

# Separate train and test data
x_train_char6Gp1count <- char6Gp1count[train$ID,] %>% as.matrix
x_test_char6Gp1count <- char6Gp1count[test$ID,] %>% as.matrix
rm(char6Gp1count,char6Gp1)

# Function to build model, make predictions and evaluate on given fold and test data
char6Gp1countModel <- function(x_train_char6Gp1count, x_test_char6Gp1count, fold, y_train){
    sai_char6Gp1count <- keras_model_sequential() %>%
        layer_dense(units = 25, activation = "relu", input_shape = ncol(x_train_char6Gp1count)) %>%
        layer_dense(units = 25, activation = "relu") %>%
        layer_dense(units = 3, activation = 'softmax')
    
    sai_char6Gp1count %>% compile(
        loss = 'categorical_crossentropy',
        optimizer = 'rmsprop',
        metrics = c('accuracy')
    )
    
    history_sai_char6Gp1count <- sai_char6Gp1count %>% fit(
        x_train_char6Gp1count[-fold,], y_train[-fold,],
        batch_size = 2^9,
        epochs = 20,
        validation_split = 0.1,
        verbose = FALSE,
        callbacks = list(
            callback_early_stopping(monitor = "val_loss", patience = 2),
            callback_model_checkpoint(
                filepath = "char6Gp1count.hdf5",
                monitor = "val_loss",
                mode = "min",
                save_best_only = TRUE)
        )
    )
    sai_char6Gp1count <- load_model_hdf5("char6Gp1count.hdf5")
    
    train_pred <- sai_char6Gp1count %>% predict(x_train_char6Gp1count[fold,])
    test_pred <- sai_char6Gp1count %>% predict(x_test_char6Gp1count)
    fold_eval <- sai_char6Gp1count %>% evaluate(x_train_char6Gp1count[fold,], y_train[fold,])
    out <- list(train_pred = train_pred, test_pred = test_pred, logloss = fold_eval$loss, acc = fold_eval$acc )
    k_clear_session()
    return(out)
}

train_char6Gp1count <- matrix(0, nrow = nrow(train), ncol = 3)
test_char6Gp1count <- matrix(0, nrow = nrow(test), ncol = 3)
metrics_char6Gp1count <- matrix(0, 5, 2)

# Form dataframes of stacked predictions, average predictions on test and evaluation metrics
for(i in 1:5){
    results_char6Gp1count <- char6Gp1countModel(x_train_char6Gp1count, x_test_char6Gp1count, 
                                                folds[[i]], y_train)
    train_char6Gp1count[folds[[i]], ] <- results_char6Gp1count$train_pred
    test_char6Gp1count <- test_char6Gp1count + (results_char6Gp1count$test_pred)/5
    metrics_char6Gp1count[i,1] <- results_char6Gp1count$logloss
    metrics_char6Gp1count[i,2] <- results_char6Gp1count$acc
}

train_char6Gp1count <- train_char6Gp1count %>% as.data.frame() %>%
    rename(char6Gp1count_EAP=V1, char6Gp1count_HPL=V2, char6Gp1count_MWS=V3)
test_char6Gp1count <- test_char6Gp1count %>% as.data.frame() %>%
    rename(char6Gp1count_EAP=V1, char6Gp1count_HPL=V2, char6Gp1count_MWS=V3)
metrics_char6Gp1count <- metrics_char6Gp1count %>% as.data.frame() %>% rename(logloss=V1, acc=V2)
rownames(metrics_char6Gp1count) <- paste0("fold ", 1:5, ":")
metrics_char6Gp1count

# char hexagrams in train for n > 15
char6Gp2 <- train %>% select(-author) %>% 
    unnest_tokens(token, text, token = "character_shingles", n=6, to_lower = FALSE, 
                  lowercase = FALSE, strip_non_alphanum = FALSE) %>%
    select(-ID) %>%
    count(token) %>%
    filter(n > 15) %>%
    .$token

# Document term matrix of char hexagrams for train and test combined for n > 15
char6Gp2count <- train %>% select(-author) %>% bind_rows(test) %>%
    unnest_tokens(token, text, token = "character_shingles", n=6, to_lower = FALSE, 
                  lowercase = FALSE, strip_non_alphanum = FALSE) %>%
    count(ID, token) %>%
    filter(token %in% char6Gp2) %>%
    cast_dtm(ID, token, n) %>%
    as.matrix()

# Append rows with zero columns in char6Gp2count for excerpts that don't contain char6Gp2
rowsRemoved <- setdiff(c(train$ID,test$ID),rownames(char6Gp2count))
allZeros <- matrix(0, length(rowsRemoved), ncol(char6Gp2count), 
                   dimnames = list(rowsRemoved, colnames(char6Gp2count)))
char6Gp2count <- char6Gp2count %>% rbind(allZeros)
rm(rowsRemoved,allZeros)

# Separate train and test data
x_train_char6Gp2count <- char6Gp2count[train$ID,] %>% as.matrix
x_test_char6Gp2count <- char6Gp2count[test$ID,] %>% as.matrix
rm(char6Gp2count,char6Gp2)

# Function to build model, make predictions and evaluate on given fold and test data
char6Gp2countModel <- function(x_train_char6Gp2count, x_test_char6Gp2count, fold, y_train){
    sai_char6Gp2count <- keras_model_sequential() %>%
        layer_dense(units = 16, activation = "relu", input_shape = ncol(x_train_char6Gp2count)) %>%
        layer_dense(units = 16, activation = "relu") %>%
        layer_dense(units = 3, activation = 'softmax')
    
    sai_char6Gp2count %>% compile(
        loss = 'categorical_crossentropy',
        optimizer = 'rmsprop',
        metrics = c('accuracy')
    )
    
    history_sai_char6Gp2count <- sai_char6Gp2count %>% fit(
        x_train_char6Gp2count[-fold,], y_train[-fold,],
        batch_size = 2^7,
        epochs = 20,
        validation_split = 0.1,
        verbose = FALSE,
        callbacks = list(
            callback_early_stopping(monitor = "val_loss", patience = 2),
            callback_model_checkpoint(
                filepath = "char6Gp2count.hdf5",
                monitor = "val_loss",
                mode = "min",
                save_best_only = TRUE)
        )
    )
    sai_char6Gp2count <- load_model_hdf5("char6Gp2count.hdf5")
    
    train_pred <- sai_char6Gp2count %>% predict(x_train_char6Gp2count[fold,])
    test_pred <- sai_char6Gp2count %>% predict(x_test_char6Gp2count)
    fold_eval <- sai_char6Gp2count %>% evaluate(x_train_char6Gp2count[fold,], y_train[fold,])
    out <- list(train_pred = train_pred, test_pred = test_pred, logloss = fold_eval$loss, acc = fold_eval$acc )
    k_clear_session()
    return(out)
}

train_char6Gp2count <- matrix(0, nrow = nrow(train), ncol = 3)
test_char6Gp2count <- matrix(0, nrow = nrow(test), ncol = 3)
metrics_char6Gp2count <- matrix(0, 5, 2)

# Form dataframes of stacked predictions, average predictions on test and evaluation metrics
for(i in 1:5){
    results_char6Gp2count <- char6Gp2countModel(x_train_char6Gp2count, x_test_char6Gp2count, 
                                                folds[[i]], y_train)
    train_char6Gp2count[folds[[i]], ] <- results_char6Gp2count$train_pred
    test_char6Gp2count <- test_char6Gp2count + (results_char6Gp2count$test_pred)/5
    metrics_char6Gp2count[i,1] <- results_char6Gp2count$logloss
    metrics_char6Gp2count[i,2] <- results_char6Gp2count$acc
}

train_char6Gp2count <- train_char6Gp2count %>% as.data.frame() %>%
    rename(char6Gp2count_EAP=V1, char6Gp2count_HPL=V2, char6Gp2count_MWS=V3)
test_char6Gp2count <- test_char6Gp2count %>% as.data.frame() %>%
    rename(char6Gp2count_EAP=V1, char6Gp2count_HPL=V2, char6Gp2count_MWS=V3)
metrics_char6Gp2count <- metrics_char6Gp2count %>% as.data.frame() %>% rename(logloss=V1, acc=V2)
rownames(metrics_char6Gp2count) <- paste0("fold ", 1:5, ":")
metrics_char6Gp2count

# From POS tag sequence for each ID
train_textTag <- train_pos_tag %>% rename(ID = doc_id) %>% 
    mutate(pos = str_replace_all(.$pos, "\\$", "S")) %>%
    group_by(ID) %>%
    summarise(textTag = str_c(pos, collapse = " "))

test_textTag <- test_pos_tag %>% rename(ID = doc_id) %>% 
    mutate(pos = str_replace_all(.$pos, "\\$", "S")) %>%
    group_by(ID) %>%
    summarise(textTag = str_c(pos, collapse = " "))


# pos tags ngrams in train
post1234G <- map_df(1:4, ~ unnest_tokens(train_textTag, token, textTag, 
                                         token = "ngrams", to_lower = FALSE, n = .x)) %>%
    count(token) %>%
    filter(n > 2) %>%
    .$token

# Document term matrix of pos tags ngrams for train and test combined
post1234Gcount <- map_df(1:4, ~ unnest_tokens(train_textTag %>% bind_rows(test_textTag), token, 
                                              textTag, token = "ngrams", to_lower = FALSE, n = .x)) %>%
    count(ID, token) %>%
    filter(token %in% post1234G) %>%
    cast_dtm(ID, token, n) %>%
    as.matrix()

# Separate train and test data
x_train_post1234Gcount <- post1234Gcount[train$ID,] %>% as.matrix
x_test_post1234Gcount <- post1234Gcount[test$ID,] %>% as.matrix
rm(post1234Gcount,post1234G)

# Function to build model, make predictions and evaluate on given fold and test data
post1234GcountModel <- function(x_train_post1234Gcount, x_test_post1234Gcount, fold, y_train){
    sai_post1234Gcount <- keras_model_sequential() %>%
        layer_dense(units = 10, activation = "relu", input_shape = ncol(x_train_post1234Gcount)) %>%
        layer_dense(units = 10, activation = "relu") %>%
        layer_dense(units = 3, activation = 'softmax')
    
    sai_post1234Gcount %>% compile(
        loss = 'categorical_crossentropy',
        optimizer = 'rmsprop',
        metrics = c('accuracy')
    )
    
    history_sai_post1234Gcount <- sai_post1234Gcount %>% fit(
        x_train_post1234Gcount[-fold,], y_train[-fold,],
        batch_size = 2^9,
        epochs = 20,
        validation_split = 0.1,
        verbose = FALSE,
        callbacks = list(
            callback_early_stopping(monitor = "val_loss", patience = 2),
            callback_model_checkpoint(
                filepath = "post1234Gcount.hdf5",
                monitor = "val_loss",
                mode = "min",
                save_best_only = TRUE)
        )
    )
    sai_post1234Gcount <- load_model_hdf5("post1234Gcount.hdf5")
    
    train_pred <- sai_post1234Gcount %>% predict(x_train_post1234Gcount[fold,])
    test_pred <- sai_post1234Gcount %>% predict(x_test_post1234Gcount)
    fold_eval <- sai_post1234Gcount %>% evaluate(x_train_post1234Gcount[fold,], y_train[fold,])
    out <- list(train_pred = train_pred, test_pred = test_pred, logloss = fold_eval$loss, acc = fold_eval$acc )
    k_clear_session()
    return(out)
}

train_post1234Gcount <- matrix(0, nrow = nrow(train), ncol = 3)
test_post1234Gcount <- matrix(0, nrow = nrow(test), ncol = 3)
metrics_post1234Gcount <- matrix(0, 5, 2)

# Form dataframes of stacked predictions, average predictions on test and evaluation metrics
for(i in 1:5){
    results_post1234Gcount <- post1234GcountModel(x_train_post1234Gcount, x_test_post1234Gcount,
                                                  folds[[i]], y_train)
    train_post1234Gcount[folds[[i]], ] <- results_post1234Gcount$train_pred
    test_post1234Gcount <- test_post1234Gcount + (results_post1234Gcount$test_pred)/5
    metrics_post1234Gcount[i,1] <- results_post1234Gcount$logloss
    metrics_post1234Gcount[i,2] <- results_post1234Gcount$acc
}

train_post1234Gcount <- train_post1234Gcount %>% as.data.frame() %>%
    rename(post1234Gcount_EAP=V1, post1234Gcount_HPL=V2, post1234Gcount_MWS=V3)
test_post1234Gcount <- test_post1234Gcount %>% as.data.frame() %>%
    rename(post1234Gcount_EAP=V1, post1234Gcount_HPL=V2, post1234Gcount_MWS=V3)
metrics_post1234Gcount <- metrics_post1234Gcount %>% as.data.frame() %>% rename(logloss=V1, acc=V2)
rownames(metrics_post1234Gcount) <- paste0("fold ", 1:5, ":")
metrics_post1234Gcount

# Word embedding and global average pooling
num_words <- 30000     # maximum number of words to consider for dictionary   

# Create dictionary
tokenizer <- text_tokenizer(num_words = num_words, lower = TRUE) %>%
    fit_text_tokenizer(train$text)  

# Convert text to sequence of integers
x_intsq_train <- texts_to_sequences(tokenizer, train$text)
x_intsq_test <- texts_to_sequences(tokenizer, test$text)

maxlen <- 100  # number of words to consider in a sequence
wv_dim <- 10   # length of word vectors
# Padding sequences
x_train_padded <- pad_sequences(x_intsq_train, maxlen = maxlen)
x_test_padded <- pad_sequences(x_intsq_test, maxlen = maxlen)

# Function to build model, make predictions and evaluate on given fold and test data
wordVecAvgModel <- function(x_train_padded, x_test_padded, fold, y_train){
    sai_wordVecAvg <- keras_model_sequential() %>%
        layer_embedding(input_dim = num_words, output_dim = wv_dim, input_length = maxlen) %>%
        layer_global_average_pooling_1d() %>%
        layer_dense(units = 50, activation = 'relu') %>%
        layer_dropout(rate = 0.1) %>%
        layer_dense(units = 3, activation = 'softmax')
    
    sai_wordVecAvg %>% compile(
        loss = 'categorical_crossentropy',
        optimizer = 'adam',
        metrics = c('accuracy')
    )
    
    history_sai_wordVecAvg <- sai_wordVecAvg %>% fit(
        x_train_padded[-fold,], y_train[-fold,],
        batch_size = 2^6, 
        epochs = 20,
        validation_split = 0.1,
        verbose = FALSE,
        callbacks = list(
            callback_early_stopping(monitor = "val_loss", patience = 2),
            callback_model_checkpoint(
                filepath = "wordVecAvg.hdf5",
                monitor = "val_loss",
                save_best_only = TRUE)
        )
    )
    
    sai_wordVecAvg <- load_model_hdf5("wordVecAvg.hdf5")
    
    train_pred <- sai_wordVecAvg %>% predict(x_train_padded[fold,])
    test_pred <- sai_wordVecAvg %>% predict(x_test_padded)
    fold_eval <- sai_wordVecAvg %>% evaluate(x_train_padded[fold,], y_train[fold,])
    out <- list(train_pred = train_pred, test_pred = test_pred, logloss = fold_eval$loss, acc = fold_eval$acc )
    k_clear_session()
    return(out)
}

train_wordVecAvg <- matrix(0, nrow = nrow(train), ncol = 3)
test_wordVecAvg <- matrix(0, nrow = nrow(test), ncol = 3)
metrics_wordVecAvg <- matrix(0, 5, 2)

# Form dataframes of stacked predictions, average predictions on test and evaluation metrics
for(i in 1:5){
    results_wordVecAvg <- wordVecAvgModel(x_train_padded, x_test_padded, folds[[i]], y_train)
    train_wordVecAvg[folds[[i]], ] <- results_wordVecAvg$train_pred
    test_wordVecAvg <- test_wordVecAvg + (results_wordVecAvg$test_pred)/5
    metrics_wordVecAvg[i,1] <- results_wordVecAvg$logloss
    metrics_wordVecAvg[i,2] <- results_wordVecAvg$acc
}

train_wordVecAvg <- train_wordVecAvg %>% as.data.frame() %>% 
    rename(wordVecAvg_EAP=V1, wordVecAvg_HPL=V2, wordVecAvg_MWS=V3)
test_wordVecAvg <- test_wordVecAvg %>% as.data.frame() %>% 
    rename(wordVecAvg_EAP=V1, wordVecAvg_HPL=V2, wordVecAvg_MWS=V3)
metrics_wordVecAvg <- metrics_wordVecAvg %>% as.data.frame() %>% rename(logloss=V1, acc=V2)
rownames(metrics_wordVecAvg) <- paste0("fold ", 1:5, ":")
metrics_wordVecAvg

# Get word pair binary correlations for words 
word_cors <- train %>% select(-author) %>% bind_rows(test) %>%
    unnest_tokens(word, text, token = "ngrams", n=1) %>% 
    group_by(word) %>%
    filter(n() >= 5) %>%    # consider words having total count more than 4
    pairwise_cor(word, ID, sort = TRUE) %>%
    cast_sparse(item1, item2, correlation)

# Get principal components on word correlation matix
pca_word_cors <- (prcomp(word_cors, center = TRUE, scale. = TRUE))$x %>% data.frame
colnames(pca_word_cors) <- paste0("wordCors", colnames(pca_word_cors))
pca_word_cors <- data_frame(word = rownames(pca_word_cors)) %>% bind_cols(pca_word_cors)

# Convert document to vector by taking mean of word vectors for each ID
meanWordCorPC <- train %>% select(-author) %>% bind_rows(test) %>%
    unnest_tokens(word, text, token = "ngrams", n=1) %>%
    count(ID, word) %>%
    bind_tf_idf(word, ID, n) %>% select(ID, word, tf_idf) %>%
    inner_join(pca_word_cors[1:1001], by = "word") %>%
    mutate_at(vars(starts_with("wordCorsPC")), funs(.*tf_idf)) %>% select(-tf_idf) %>%
    group_by(ID) %>%
    summarise_at(vars(starts_with("wordCorsPC")), mean)

# Separate train and test data
x_train_meanWordCorPC <- train[,"ID"] %>%
    left_join(meanWordCorPC, by = "ID") 
x_train_meanWordCorPC[is.na(x_train_meanWordCorPC)] <- 0

x_test_meanWordCorPC <- test[,"ID"] %>%
    left_join(meanWordCorPC, by = "ID") 
x_test_meanWordCorPC[is.na(x_test_meanWordCorPC)] <- 0

x_train_meanWordCorPC <- x_train_meanWordCorPC %>% select(-ID) %>% as.matrix()
x_test_meanWordCorPC <- x_test_meanWordCorPC %>% select(-ID) %>% as.matrix()

rm(meanWordCorPC,pca_word_cors,word_cors)

# Function to build model, make predictions and evaluate on given fold and test data
meanWordCorPCModel <- function(x_train_meanWordCorPC, x_test_meanWordCorPC, fold, y_train){
    sai_meanWordCorPC <- keras_model_sequential() %>%
        layer_dense(units = 25, activation = "relu", input_shape = ncol(x_train_meanWordCorPC)) %>%
        layer_dropout(rate = 0.01) %>%
        layer_dense(units = 25, activation = "relu") %>%
        layer_dropout(rate = 0.01) %>%
        layer_dense(units = 20, activation = "relu") %>%
        layer_dropout(rate = 0.01) %>%
        layer_dense(units = 20, activation = "relu") %>%
        layer_dropout(rate = 0.01) %>%
        layer_dense(units = 20, activation = "relu") %>%
        layer_dropout(rate = 0.01) %>%
        layer_dense(units = 3, activation = 'softmax')
    
    sai_meanWordCorPC %>% compile(
        loss = 'categorical_crossentropy',
        optimizer = 'adam',
        metrics = c('accuracy')
    )
    
    history_sai_meanWordCorPC <- sai_meanWordCorPC %>% fit(
        x_train_meanWordCorPC[-fold,], y_train[-fold,],
        batch_size = 2^5,
        epochs = 20,
        validation_split = 0.1,
        verbose = FALSE,
        callbacks = list(
            callback_early_stopping(monitor = "val_loss", patience = 2),
            callback_model_checkpoint(
                filepath = "meanWordCorPC.hdf5",
                monitor = "val_loss",
                mode = "min",
                save_best_only = TRUE)
        )
    )
    sai_meanWordCorPC <- load_model_hdf5("meanWordCorPC.hdf5")
    
    train_pred <- sai_meanWordCorPC %>% predict(x_train_meanWordCorPC[fold,])
    test_pred <- sai_meanWordCorPC %>% predict(x_test_meanWordCorPC)
    fold_eval <- sai_meanWordCorPC %>% evaluate(x_train_meanWordCorPC[fold,], y_train[fold,])
    out <- list(train_pred = train_pred, test_pred = test_pred, logloss = fold_eval$loss, acc = fold_eval$acc )
    k_clear_session()
    return(out)
}

train_meanWordCorPC <- matrix(0, nrow = nrow(train), ncol = 3)
test_meanWordCorPC <- matrix(0, nrow = nrow(test), ncol = 3)
metrics_meanWordCorPC <- matrix(0, 5, 2)

# Form dataframes of stacked predictions, average predictions on test and evaluation metrics
for(i in 1:5){
    results_meanWordCorPC <- meanWordCorPCModel(x_train_meanWordCorPC, x_test_meanWordCorPC, 
                                                folds[[i]], y_train)
    train_meanWordCorPC[folds[[i]], ] <- results_meanWordCorPC$train_pred
    test_meanWordCorPC <- test_meanWordCorPC + (results_meanWordCorPC$test_pred)/5
    metrics_meanWordCorPC[i,1] <- results_meanWordCorPC$logloss
    metrics_meanWordCorPC[i,2] <- results_meanWordCorPC$acc
}

train_meanWordCorPC <- train_meanWordCorPC %>% as.data.frame() %>%
    rename(meanWordCorPC_EAP=V1, meanWordCorPC_HPL=V2, meanWordCorPC_MWS=V3)
test_meanWordCorPC <- test_meanWordCorPC %>% as.data.frame() %>%
    rename(meanWordCorPC_EAP=V1, meanWordCorPC_HPL=V2, meanWordCorPC_MWS=V3)
metrics_meanWordCorPC <- metrics_meanWordCorPC %>% as.data.frame() %>% rename(logloss=V1, acc=V2)
rownames(metrics_meanWordCorPC) <- paste0("fold ", 1:5, ":")
metrics_meanWordCorPC

# Get word pair pointwise mutual information for words 
word_pmi <- train %>% select(-author) %>% bind_rows(test) %>%
    unnest_tokens(word, text, token = "ngrams", n=1) %>% 
    group_by(word) %>%
    filter(n() >= 5) %>%
    pairwise_pmi(word, ID, sort = TRUE) %>%
    cast_sparse(item1, item2, pmi)

# Get principal components on word pmi matix
pca_word_pmi <- (fast.prcomp(word_pmi, center = TRUE, scale. = TRUE))$x %>% data.frame
colnames(pca_word_pmi) <- paste0("wordPmi", colnames(pca_word_pmi))
pca_word_pmi <- data_frame(word = rownames(word_pmi)) %>% bind_cols(pca_word_pmi)

# Convert document to vector by taking mean of word vectors for each ID
meanWordPmiPC <- train %>% select(-author) %>% bind_rows(test) %>%
    unnest_tokens(word, text, token = "ngrams", n=1) %>%
    count(ID, word) %>%
    bind_tf_idf(word, ID, n) %>% select(ID, word, tf_idf) %>%
    inner_join(pca_word_pmi[1:1001], by = "word") %>%
    mutate_at(vars(starts_with("wordPmiPC")), funs(.*tf_idf)) %>% select(-tf_idf) %>%
    group_by(ID) %>%
    summarise_at(vars(starts_with("wordPmiPC")), mean)

# Separate train and test data
x_train_meanWordPmiPC <- train[,"ID"] %>%
    left_join(meanWordPmiPC, by = "ID") 
x_train_meanWordPmiPC[is.na(x_train_meanWordPmiPC)] <- 0

x_test_meanWordPmiPC <- test[,"ID"] %>%
    left_join(meanWordPmiPC, by = "ID") 
x_test_meanWordPmiPC[is.na(x_test_meanWordPmiPC)] <- 0
rm(meanWordPmiPC,pca_word_pmi,word_pmi)

x_train_meanWordPmiPC <- x_train_meanWordPmiPC %>% select(-ID) %>% as.matrix()
x_test_meanWordPmiPC <- x_test_meanWordPmiPC %>% select(-ID) %>% as.matrix()

# Function to build model, make predictions and evaluate on given fold and test data
meanWordPmiPCModel <- function(x_train_meanWordPmiPC, x_test_meanWordPmiPC, fold, y_train){
    sai_meanWordPmiPC <- keras_model_sequential() %>%
        layer_dense(units = 25, activation = "relu", input_shape = ncol(x_train_meanWordPmiPC)) %>%
        layer_dropout(rate = 0.01) %>%
        layer_dense(units = 25, activation = "relu") %>%
        layer_dropout(rate = 0.01) %>%
        layer_dense(units = 20, activation = "relu") %>%
        layer_dropout(rate = 0.01) %>%
        layer_dense(units = 20, activation = "relu") %>%
        layer_dropout(rate = 0.01) %>%
        layer_dense(units = 20, activation = "relu") %>%
        layer_dropout(rate = 0.01) %>%
        layer_dense(units = 3, activation = 'softmax')
    
    sai_meanWordPmiPC %>% compile(
        loss = 'categorical_crossentropy',
        optimizer = 'adam',
        metrics = c('accuracy')
    )
    
    history_sai_meanWordPmiPC <- sai_meanWordPmiPC %>% fit(
        x_train_meanWordPmiPC[-fold,], y_train[-fold,],
        batch_size = 2^5,
        epochs = 20,
        validation_split = 0.1,
        verbose = FALSE,
        callbacks = list(
            callback_early_stopping(monitor = "val_loss", patience = 2),
            callback_model_checkpoint(
                filepath = "meanWordPmiPC.hdf5",
                monitor = "val_loss",
                mode = "min",
                save_best_only = TRUE)
        )
    )
    sai_meanWordPmiPC <- load_model_hdf5("meanWordPmiPC.hdf5")
    
    train_pred <- sai_meanWordPmiPC %>% predict(x_train_meanWordPmiPC[fold,])
    test_pred <- sai_meanWordPmiPC %>% predict(x_test_meanWordPmiPC)
    fold_eval <- sai_meanWordPmiPC %>% evaluate(x_train_meanWordPmiPC[fold,], y_train[fold,])
    out <- list(train_pred = train_pred, test_pred = test_pred, logloss = fold_eval$loss, acc = fold_eval$acc )
    k_clear_session()
    return(out)
}

train_meanWordPmiPC <- matrix(0, nrow = nrow(train), ncol = 3)
test_meanWordPmiPC <- matrix(0, nrow = nrow(test), ncol = 3)
metrics_meanWordPmiPC <- matrix(0, 5, 2)

# Form dataframes of stacked predictions, average predictions on test and evaluation metrics
for(i in 1:5){
    results_meanWordPmiPC <- meanWordPmiPCModel(x_train_meanWordPmiPC, x_test_meanWordPmiPC, 
                                                folds[[i]], y_train)
    train_meanWordPmiPC[folds[[i]], ] <- results_meanWordPmiPC$train_pred
    test_meanWordPmiPC <- test_meanWordPmiPC + (results_meanWordPmiPC$test_pred)/5
    metrics_meanWordPmiPC[i,1] <- results_meanWordPmiPC$logloss
    metrics_meanWordPmiPC[i,2] <- results_meanWordPmiPC$acc
}

train_meanWordPmiPC <- train_meanWordPmiPC %>% as.data.frame() %>%
    rename(meanWordPmiPC_EAP=V1, meanWordPmiPC_HPL=V2, meanWordPmiPC_MWS=V3)
test_meanWordPmiPC <- test_meanWordPmiPC %>% as.data.frame() %>%
    rename(meanWordPmiPC_EAP=V1, meanWordPmiPC_HPL=V2, meanWordPmiPC_MWS=V3)
metrics_meanWordPmiPC <- metrics_meanWordPmiPC %>% as.data.frame() %>% rename(logloss=V1, acc=V2)
rownames(metrics_meanWordPmiPC) <- paste0("fold ", 1:5, ":")
metrics_meanWordPmiPC

# Consolidating all stacked predictions
train_all_sp <- train %>% select(-text) %>% 
    bind_cols(train_word1Gcount) %>%
    bind_cols(train_word2Gcount) %>%
    bind_cols(train_word3Gcount) %>%
    bind_cols(train_word4Gcount) %>%
    bind_cols(train_char123Gcount) %>%
    bind_cols(train_char4Gcount) %>%
    bind_cols(train_char5Gp1count) %>%
    bind_cols(train_char5Gp2count) %>%
    bind_cols(train_char6Gp1count) %>%
    bind_cols(train_char6Gp2count) %>%
    bind_cols(train_post1234Gcount) %>%
    bind_cols(train_wordVecAvg) %>%
    bind_cols(train_meanWordCorPC) %>%
    bind_cols(train_meanWordPmiPC)

test_all_sp <- test %>% select(-text) %>% 
    bind_cols(test_word1Gcount) %>%
    bind_cols(test_word2Gcount) %>%
    bind_cols(test_word3Gcount) %>%
    bind_cols(test_word4Gcount) %>%
    bind_cols(test_char123Gcount) %>%
    bind_cols(test_char4Gcount) %>%
    bind_cols(test_char5Gp1count) %>%
    bind_cols(test_char5Gp2count) %>%
    bind_cols(test_char6Gp1count) %>%
    bind_cols(test_char6Gp2count) %>%
    bind_cols(test_post1234Gcount) %>%
    bind_cols(test_wordVecAvg) %>%
    bind_cols(test_meanWordCorPC) %>%
    bind_cols(test_meanWordPmiPC)

sp_plot <- train_all_sp %>%
    select(-ID) %>%
    group_by(author) %>%
    summarise_all(mean) %>%
    gather(feature, value, -author) %>%
    ggplot(aes(x = feature, y = value, color = author, fill = author, size = value)) +
    geom_point(alpha = 0.6) +
    labs(x = "Feature", y = "Mean Value by Author") +
    coord_flip()
ggplotly(sp_plot, tooltip = c("x","y","fill"))

# Consolidating all features for training and testing
dtrain <- train %>% select(-text) %>% 
    left_join(train_pos_count, by = "ID") %>%
    left_join(train_upos_count, by = "ID") %>%
    left_join(train_author_pair_only, by = "ID") %>%
    left_join(train_author_only, by = "ID") %>%
    left_join(train_senti, by = "ID") %>%
    left_join(train_stylo, by = "ID") %>%
    left_join(train_all_sp, by = c("ID", "author"))
dtrain[is.na(dtrain)] <- 0

dtest <- test %>% select(-text) %>% 
    left_join(test_pos_count, by = "ID") %>%
    left_join(test_upos_count, by = "ID") %>%
    left_join(test_author_pair_only, by = "ID") %>%
    left_join(test_author_only, by = "ID") %>%
    left_join(test_senti, by = "ID") %>%
    left_join(test_stylo, by = "ID") %>%
    left_join(test_all_sp, by = "ID") 
dtest[is.na(dtest)] <- 0

dim(dtrain)

dim(dtest)

# Spearman correlations between covariates
sCorr <- cor(dtrain %>% select(-ID, -author), method = "spearman")
sigCorr <- findCorrelation(sCorr, cutoff = .8, names = TRUE)
ggcorrplot(cor(dtrain %>% select(sigCorr), method = "spearman"))

# One-way analysis of variance 
anova <- apply(dtrain %>% select(-ID,-author), 2,
               function(x, y)
               {
                   lm_model <- lm(x ~ y)
                   unlist(glance(lm_model)[c("r.squared", "p.value")])
               },
               y = dtrain$author)
anova <- as.data.frame(t(anova))
names(anova) <- c("r.squared", "p.value")
anova$p.adj <- p.adjust(anova$p.value , method = "fdr")
anova$predictor <- rownames(anova)

anova_plot <- anova %>% mutate(grp = case_when(
    p.adj == 0 ~ "p.adj==0",
    p.adj > 0 & p.adj < 0.01 ~ "0<p.adj<0.01",
    p.adj >= 0.01 ~ "p.adj>=0.01")) %>%
    ggplot(aes(x=grp, y=r.squared, color=grp)) +
    geom_jitter(aes(text = (paste("Predictor:", predictor, 
                                  "<br>p.adj:", p.adj,
                                  "<br>R squared:", r.squared))), 
                alpha = 0.4, size = 3, height = 0) +
    labs(x = "Features grouped by", y = "Coefficient of determination") +
    theme(legend.position = "none")
ggplotly(anova_plot, tooltip = c("text"))

# Features having high correlations
(highCorrVars <- findCorrelation(sCorr, cutoff = .98, names = TRUE))

# Feature selection at 0.01 significance level
sig_.01_f <- anova %>% filter(p.adj < 0.01) %>% .$predictor

# Function for measuring accuracy and log loss
customSummary <- function(data, lev = levels(data$obs), model = NULL) {
    mcs <- multiClassSummary(data, lev = levels(data$obs), model = NULL)
    mnll <- mnLogLoss(data, lev = levels(data$obs), model = NULL)
    out <- c(mnll, mcs['Accuracy'])
}

# Xgboost model
set.seed(3612)
sai_xgbt <- train(x = dtrain %>% select(setdiff(sig_.01_f, highCorrVars)),
                  y = factor(dtrain$author),
                  method = "xgbTree",
                  metric = "logLoss",
                  tuneGrid = expand.grid(nrounds = seq(50,1400,50),
                                         max_depth = 4, 
                                         eta = 0.02,
                                         gamma = 0.5,
                                         colsample_bytree = 0.35,
                                         min_child_weight = 4,
                                         subsample = 0.85),
                  trControl = trainControl(method = "cv",
                                           number = 10,
                                           classProbs = TRUE,
                                           summaryFunction = customSummary,
                                           search = "grid")
)
sai_xgbt

sub_sai_xgbt <- read_csv("data/sample_submission.csv") %>% select(id) %>%
    bind_cols(predict(sai_xgbt, dtest, type = "prob"))
write_excel_csv(sub_sai_xgbt, "stacked_xgb.csv.csv")
