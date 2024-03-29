# Generated by Django 4.2.5 on 2024-02-16 10:09

import datetime
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0011_annotator_alter_basemodel_timestamp_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='basemodel',
            name='timestamp',
            field=models.DateTimeField(blank=True, default=datetime.datetime(2024, 2, 16, 15, 39, 26, 686732)),
        ),
        migrations.AlterField(
            model_name='fabric',
            name='timeStamp',
            field=models.DateTimeField(blank=True, default=datetime.datetime(2024, 2, 16, 15, 39, 26, 686732)),
        ),
        migrations.CreateModel(
            name='Tasks',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('datasetName', models.CharField(blank=True, default='', max_length=100)),
                ('tasks', models.JSONField(blank=True, default=dict)),
                ('fabric', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='api.fabric')),
            ],
        ),
    ]
