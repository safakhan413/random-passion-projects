import mysql.connector
from credentials import credential


con = mysql.connector.connect(
    user = credential['user'],
    password = credential['password'],
    host = credential['host'],
    database = credential['database']
)

cursor = con.cursor()
word = input("Enter a word:")
query = cursor.execute("SELECT * FROM Dictionary where Expression = '%s' " % word)
results = cursor.fetchall()
# print(results[0])
if results:
    for result in results:
        print(result[1])
else:
    print('No results found')