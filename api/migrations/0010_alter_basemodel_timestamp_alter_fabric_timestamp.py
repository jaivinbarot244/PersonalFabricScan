# Generated by Django 4.2.5 on 2024-02-15 05:15

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0009_alter_basemodel_timestamp_alter_fabric_timestamp_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='basemodel',
            name='timestamp',
            field=models.DateTimeField(blank=True, default=datetime.datetime(2024, 2, 15, 10, 45, 34, 510331)),
        ),
        migrations.AlterField(
            model_name='fabric',
            name='timeStamp',
            field=models.DateTimeField(blank=True, default=datetime.datetime(2024, 2, 15, 10, 45, 34, 511329)),
        ),
    ]
