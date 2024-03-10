# Generated by Django 4.2.5 on 2024-02-16 10:12

import datetime
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0012_alter_basemodel_timestamp_alter_fabric_timestamp_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='basemodel',
            name='timestamp',
            field=models.DateTimeField(blank=True, default=datetime.datetime(2024, 2, 16, 15, 42, 52, 294764)),
        ),
        migrations.AlterField(
            model_name='fabric',
            name='timeStamp',
            field=models.DateTimeField(blank=True, default=datetime.datetime(2024, 2, 16, 15, 42, 52, 294764)),
        ),
        migrations.AlterField(
            model_name='tasks',
            name='datasetName',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='api.datasets'),
        ),
    ]