# Generated by Django 4.2.5 on 2024-02-10 13:49

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Basemodel',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('modelName', models.CharField(blank=True, default='', max_length=100)),
                ('modelType', models.CharField(blank=True, default='', max_length=100)),
            ],
        ),
    ]