from rest_framework import serializers
from thesaurus.models import Thesuarusitem

class ThesaurusSerializer(serializers.ModelSerializer):
    class Meta:
        # managed = False
        model = Thesuarusitem
        db_table = 'data'
        fields = ('word', 'meaning')
