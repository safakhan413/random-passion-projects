from rest_framework import serializers
from thesaurus.models import Thesuarusitem

class ThesaurusSerializer(serializers.ModelSerializer):
    class Meta:
        # managed = False
        model = Thesuarusitem
        db_table = "thesaurus_thesuarusitem"
        fields = ('id, ''word', 'meaning')
