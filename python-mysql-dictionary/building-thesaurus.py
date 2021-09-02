import difflib
from difflib import SequenceMatcher, get_close_matches
import json
import urllib.request
import sqlite3
import os.path

### Fuzzy matching ##################
fname = 'data.json'
data = json.load(open(fname))

# print(data)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(BASE_DIR, "Dictionary.db")
print(db_path)
connection = sqlite3.connect(db_path, timeout=10)
# connection.close()
cursor = connection.cursor()

# cursor.execute('''CREATE TABLE data(
# word VARCHAR NOT NULL,
# meaning VARCHAR NOT NULL)''')
# # # connection = sqlite3.connect('file:dictionary.db?mode=rw', uri=True)
# # cursor = connection.cursor()
# #
# # print(os.getcwd())
# query = "Insert into data(word, meaning) values (?, ?)"
# cursor.execute(query, ('safa', 'khan'))

# connection.commit()
# cursor.close()
# connection.close()
# connection.close()
p = list(data.keys())
q = list(data.values())
print(list(p)[0])
print(q[0])
# print(len(p))
#
for i in range(len(p)):
    query = "Insert into data(word, meaning) values (?, ?)"
    for meanings in q[i]:
        cursor.execute(query, (p[i], meanings))
        connection.commit()

cursor.close()
connection.close()
# cursor.executemany("Insert into dictionary values (?, ?)",
#                        (data.keys(), child['close'], child['high'], child['low'],
#                        child['open'], child['volume'], stock))
#         connection.commit()

# for child in data:
#     print(child)

# def translate(w):
#     word = w.lower()
#     if word in data:
#         return data[word]
#     elif len(get_close_matches(word, data.keys(), n = 1, cutoff=0.8))>0 :
#         close_match = get_close_matches(word, data.keys(), n = 1, cutoff=0.8)
#         print( "Did you mean %s instead? " % close_match[0])
#         response = input("Answer y/n (Note input is case sensitive)")
#         if response == 'y':
#             return data[close_match[0]]
#         elif response == 'n':
#             return "Sorry, word doesn't exist.Please check again."
#         else:
#             return "We didn't understand your entry"
#     else:
#         return "The word doesn't exist. Please double check it."
#
# word = input('Enter word: ')
# output = translate(word)
# if type(output) == list:
#     for item in output:
#         print(item)
# else:
#     print(output)