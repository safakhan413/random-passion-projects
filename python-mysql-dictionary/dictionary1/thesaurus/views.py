from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import difflib
from difflib import SequenceMatcher, get_close_matches

# from rest_framework.parsers import JSONParser
# from django.http.response import JsonResponse
import sqlite3


from thesaurus.models import Thesuarusitem
from thesaurus.serializers import ThesaurusSerializer
#
# from django.core.files.storage import default_storage
# Create your views here.
def thesaurusView(request):
    # return render(request, 'Heyloo')

    return render(request, 'thesaurus.html')


def word(request):
    word = request.GET['word']
    word = word.lower() ## to make sure that function is lower and upper case insensitive

    print('###############',word)
    words = Thesuarusitem.objects.all()
    word_query = words.raw("SELECT id, word FROM thesaurus_thesuarusitem ")
    worditems = list()
    meaning = list()

    for w in word_query:
        worditems.append(w.word)
    # if word in words.raw("SELECT id, word FROM thesaurus_thesuarusitem WHERE word = '%s'" %word).word:
    raw_query_results = words.raw("SELECT id, meaning FROM thesaurus_thesuarusitem WHERE word = '%s'" %word)
    if len(list(raw_query_results)) > 0:

        for w in raw_query_results:
            # if len(list())
            # meaning = meaning + '\n' + w.meaning
            meaning.append(w.meaning)
            print('Im all words tables $*^*&^&*^',meaning)
        results = {
            'word': word,
            'meaning': meaning,
        }

        return render(request, 'word.html', {'results': results})
    else:

        # print('!##$$$$$$$$$$$$$,', worditems)
        close_match = get_close_matches(word, worditems, n = 1, cutoff=0.8)
        print('!##$$$$$$$$$$$$$ im closematch,', close_match)
        results = {
            'word': word,
            'meaning': ["We don't have this word in dictionary. Did you mean '%s'. If yes please input this word again to search." %close_match[0]],
        }
        return render(request, 'word.html', {'results': results})
    # else:
    # results = {
    #     'word': word,
    #     'meaning': meaning,
    # }
    #
    # return render(request, 'word.html', {'results': results})

    # query =
    # thesaurus_serializer = ThesaurusSerializer(words, many=True)
    # print('Im words###################', words[0].word)



    # return render(request, 'word.html')


    # res = requests.get('https://www.dictionary.com/browse/' + word)
    # res2 = requests.get('https://www.thesaurus.com/browse/' + word)
    #
    # if res:
    #     soup = bs4.BeautifulSoup(res.text, 'lxml')
    #
    #     meaning = soup.find_all('div', {'value': '1'})
    #     meaning1 = meaning[0].getText()
    # else:
    #     word = 'Sorry, ' + word + ' Is Not Found In Our Database'
    #     meaning = ''
    #     meaning1 = ''
    #
    # if res2:
    #     soup2 = bs4.BeautifulSoup(res2.text, 'lxml')
    #
    #     synonyms = soup2.find_all('a', {'class': 'css-r5sw71-ItemAnchor etbu2a31'})
    #     ss = []
    #     for b in synonyms[0:]:
    #         re = b.text.strip()
    #         ss.append(re)
    #     se = ss
    #
    #     antonyms = soup2.find_all('a', {'class': 'css-lqr09m-ItemAnchor etbu2a31'})
    #     aa = []
    #     for c in antonyms[0:]:
    #         r = c.text.strip()
    #         aa.append(r)
    #     ae = aa
    # else:
    #     se = ''
    #     ae = ''
    #
    # results = {
    #     'word': word,
    #     'meaning': meaning1,
    # }
    #
    # return render(request, 'word.htm', {'se': se, 'ae': ae, 'results': results})

# export DJANGO_SETTINGS_MODULE=dictionary.settings
