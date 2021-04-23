# ref https://towardsdatascience.com/generating-text-using-a-recurrent-neural-network-1c3bfee27a5e
import numpy as np
## Creating our dataset
with open('sherlock_homes.txt', 'r') as file:
    text = file.read().lower()
# print('text length', len(text))
# print('text length', text)
chars = sorted(list(set(text))) ## get only unique characters
# print('total chars: ', len(chars))
# print('total chars: ', chars)
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
# print(indices_char)

## split data into subsequenes of length 40 each and then transform data to a boolean array
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen]) ## 40 chracters preceding the next char. Step 3 chars
    next_chars.append(text[i + maxlen]) # next char after 40 characters of step 3 each

# print(f'im sentence {sentences[0:3]}')
# print(f'im text chars {text[40:43]}')
# print(f'im nextchars {next_chars[0:3]}')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)




