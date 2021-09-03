from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
# from rest_framework.parsers import JSONParser
# from django.http.response import JsonResponse

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
    print('###############',word)
    words = type(Thesuarusitem)
    thesaurus_serializer = ThesaurusSerializer(words, many=True)
    print('Im words###################', words)
    return render(request, 'word.html')

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
