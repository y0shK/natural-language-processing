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

