# Generated by Django 3.1.2 on 2021-09-02 22:11

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('thesaurus', '0006_authgroup_authgrouppermissions_authpermission_authuser_authusergroups_authuseruserpermissions_django'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='thesuarusitem',
            options={'managed': True},
        ),
        migrations.AlterField(
            model_name='thesuarusitem',
            name='id',
            field=models.AutoField(primary_key=True, serialize=False),
        ),
        migrations.AlterModelTable(
            name='thesuarusitem',
            table='thesaurus_thesuarusitem',
        ),
    ]