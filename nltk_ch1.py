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
