# use Harry Potter .txt data to find patterns
import nltk
import os

# needs absolute path - can't provide ~
os.chdir('/home/yash/harry_potter')

# https://stackoverflow.com/questions/10467024/how-do-i-create-my-own-nltk-text-from-a-text-file

f_hp7 = open('Book 7 - The Deathly Hallows.txt','r') # open the text
raw_hp7 = f_hp7.read() # read the text
tokens_hp7 = nltk.word_tokenize(raw_hp7) # take the entire raw string and divide it into a list substrings

print(tokens_hp7)

text_hp7 = nltk.Text(tokens_hp7) # converts list of substrings to an nltk data type - now can use concordance(), etc.

print(text_hp7)

print(type(tokens_hp7))
print(type(text_hp7))

hp7 = text_hp7

hp7.concordance('Horcrux')
hp7.similar('Horcrux')

hp7.concordance('love')
hp7.similar('love')

print(hp7.count('Horcrux'))
print(hp7.count('Voldemort'))
print(hp7.count('love'))
print(hp7.count('soul'))

hp7_len = len(hp7)
hp7_sets_of_words = sorted(set(hp7))
print(hp7_len)
print(hp7_sets_of_words)

# Deathly Hallows lexical diversity
hp7_lexical_diversity = len(hp7_sets_of_words) / hp7_len # lexical diversity = different types of words / total words
print(hp7_lexical_diversity)

# Deathly Hallows statistics
hp7_lowercase_alphas = [word.lower() for word in hp7 if word.isalpha()]
hp7_lowercase_alphas = [word for word in hp7_lowercase_alphas if word != 'rowling'] # remove author's name from text

print(sorted(word for word in hp7_lowercase_alphas if word.startswith('pat')))
print(sorted(word for word in hp7_lowercase_alphas if word.endswith('ness')))
print(sorted(word for word in hp7_lowercase_alphas if 'min' in word))

hp7_complex_words = [word for word in hp7_lowercase_alphas if len(word) > 2 and word != 'the' and word != 'and']

hp7_longer_words = [word for word in hp7_lowercase_alphas if len(word) > 4]

hp7_freqDist_complex = nltk.FreqDist(hp7_complex_words)
hp7_freqDist_complex.plot(25, cumulative=True)
hp7_freqDist_complex.plot(25, cumulative=False)

hp7_freqDist_longer = nltk.FreqDist(hp7_longer_words)
hp7_freqDist_longer.plot(25, cumulative=True)
hp7_freqDist_longer.plot(25, cumulative=False)

# apply some techniques from Ch. 2 - accessing corpora and lexical resources
houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
house_concordance_list = []

for house in houses:
    house_concordance_amt = hp7.concordance(house)

    house_concordance_list.append(house_concordance_amt)
    print(house + ':', hp7.count(house))
print(house_concordance_list)

# split DH into two parts, access House mentions by part
hp7_part1 = hp7[0:len(hp7)//2]
hp7_part2 = hp7[len(hp7)//2:hp7_len]

hp7_part1 = nltk.Text(hp7_part1) # convert to nltk.Text to use concordance()
hp7_part2 = nltk.Text(hp7_part2)

# for each Hogwarts House, search for its occurrence in text
for house in houses:
    house_concordance_part1 = hp7_part1.concordance(house)
    house_concordance_part2 = hp7_part2.concordance(house)

print('\n')

# https://stackoverflow.com/questions/32676319/new-to-nltk-having-trouble-with-conditional-frequency
# useful resource for freq and conditional freq distribution

fDistHouses = nltk.FreqDist(tokens_hp7) # take list of substrings of HP text
for word in fDistHouses:
    if word in houses:
        print('Frequency of', word, fDistHouses.freq(word))

print("\n")

interesting_words = ['Horcrux', 'Hallow', 'Dumbledore', 'love', 'Patronus', 'Ministry', 'Voldemort', 'Weasley', 'wand',
                     'Snape', 'Crucio', 'Muggle', 'Harry']

fDistInterestingWords = nltk.FreqDist(tokens_hp7)
for word in fDistInterestingWords:
    if word in interesting_words:
        print('Frequency of', word, fDistInterestingWords.freq(word))

print('\n')

# some words have a large number of letters - unique in n-letter length
# no point in using these unique n-letter lengths in a conditional distribution
interesting_words.remove('Dumbledore')
interesting_words.remove('Voldemort')

conditional_freq_dist_interesting_words = nltk.ConditionalFreqDist()

for word in tokens_hp7:
    if word in interesting_words:
        condition = len(word)
        conditional_freq_dist_interesting_words[condition][word] += 1

for condition in conditional_freq_dist_interesting_words:
    for word in conditional_freq_dist_interesting_words[condition]:
        print("Conditional frequency of", word, conditional_freq_dist_interesting_words[condition].freq(word), "condition = word length", condition)

hp7_string_text = tokens_hp7
list(nltk.bigrams(hp7_string_text))

def generate_model(cfd, word):
    for i in range(15):
        print(word, end=' ')
        word = cfd[word].max()

bigrams = nltk.bigrams(hp7_string_text)
cfd_bigrams = nltk.ConditionalFreqDist(bigrams)
generate_model(cfd_bigrams, 'Horcrux')

from nltk.sentiment.vader import SentimentIntensityAnalyzer
tokens_sentences_hp7 = nltk.sent_tokenize(raw_hp7)

# remove specific substrings - https://stackoverflow.com/questions/37372603/how-to-remove-specific-substrings-from-a-set-of-strings-in-python

# Vader explanation: https://stackoverflow.com/questions/40325980/how-is-the-vader-compound-polarity-score-calculated-in-python-nltk
# Vader source code: https://github.com/nltk/nltk/blob/develop/nltk/sentiment/vader.py

formatted_tokens_sentences_hp7 = [x.replace('\n', '') for x in tokens_sentences_hp7] # list comprehension to replace newline formatting
print(formatted_tokens_sentences_hp7)

# create instance for vader algorithm
sent_analyzer = SentimentIntensityAnalyzer()

for sent in formatted_tokens_sentences_hp7:
    print(sent) # each sentence of the book, sentiment analysis performed on each by vader
    polarity = sent_analyzer.polarity_scores(sent) # polarity_scores finds the overall "sentiment" by looking at all of the words and quantifying them
    for k in sorted(polarity):
        print('{0}: {1}, '.format(k, polarity[k]), '\n\n')
