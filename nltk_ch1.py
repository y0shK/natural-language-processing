from nltk.book import *
import nltk

genesis_text = text3
monty_python_text = text6

genesis_text.concordance('light')
genesis_text.similar('light')

monty_python_text.concordance('sword')
monty_python_text.similar('sword')

inaugural_address_text = text4
inaugural_address_text.dispersion_plot(['democracy', 'freedom'])

print("\nGenerate\n")

inaugural_address_text.generate()

# intuition for diversity of words
monty_python_len = len(monty_python_text)
print(monty_python_len)
print(len(inaugural_address_text))

monty_python_sets_of_words = sorted(set(monty_python_text))

print(monty_python_sets_of_words)

# lexical diversity - how many different words are used as compared to how many words are used in total
monty_python_lexical_diversity = len(monty_python_sets_of_words) / monty_python_len
print('\nLexical diversity: ' + str(monty_python_lexical_diversity))

# find amount of times a word is used on average
monty_python_lexical_words = monty_python_lexical_diversity * len(monty_python_text)
print('Each word is used about ' + str(monty_python_lexical_words) + ' times on average')

inaugural_address_len = len(inaugural_address_text)
inaugural_address_sets_of_words = sorted(set(inaugural_address_text))

inaugural_address_lexical_diversity = len(inaugural_address_sets_of_words) / inaugural_address_len
print('\nLexical diversity: ' + str(inaugural_address_lexical_diversity))

inaugural_address_lexical_words = inaugural_address_lexical_diversity * len(inaugural_address_text)
print('Each word is used about ' + str(inaugural_address_lexical_words) + ' times on average' + '\n')

democracy_count = inaugural_address_text.count('democracy')
america_count = inaugural_address_text.count('America')

print(democracy_count)
print(str(america_count) + '\n')

half_of_democracy_speech = inaugural_address_text[0:inaugural_address_len//2]
other_half_of_democracy_speech = inaugural_address_text[inaugural_address_len//2:inaugural_address_len]

first_half = half_of_democracy_speech.count('democracy')
second_half = other_half_of_democracy_speech.count('democracy')

print(first_half)
print(second_half)
print(half_of_democracy_speech.index('democracy'))
print(other_half_of_democracy_speech.index('democracy'))
print('\n')

# section 3 - computing language with statistics
inaugural_address_frequency_distribution = nltk.FreqDist(inaugural_address_text)
print(inaugural_address_frequency_distribution['freedom']) # how many times does the word show up?

# first parameter is how many words are added
inaugural_address_frequency_distribution.plot(75, cumulative=True)

# borrow concepts from set theory - look at longest words
inaugural_set = set(inaugural_address_text)
long_inaugural_words = [word for word in inaugural_set if len(word) > 10]
long_inaugural_words = sorted(long_inaugural_words)

print(long_inaugural_words)

