# Generated by Django 4.2.5 on 2024-03-11 05:34

import datetime
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0016_alter_basemodel_timestamp_alter_fabric_timestamp'),
    ]

    operations = [
        migrations.AlterField(
            model_name='basemodel',
            name='timestamp',
            field=models.DateTimeField(blank=True, default=datetime.datetime(2024, 3, 11, 11, 4, 34, 388382)),
        ),
        migrations.AlterField(
            model_name='fabric',
            name='timeStamp',
            field=models.DateTimeField(blank=True, default=datetime.datetime(2024, 3, 11, 11, 4, 34, 388382)),
        ),
        migrations.CreateModel(
            name='Yamlfile',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('yamlPath', models.CharField(blank=True, default='', max_length=500)),
                ('dataset', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='api.datasets')),
                ('fabric', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='api.fabric')),
            ],
        ),
        migrations.CreateModel(
            name='Label',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('label', models.CharField(blank=True, default='', max_length=500)),
                ('dataset', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='api.datasets')),
                ('fabric', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='api.fabric')),
                ('image', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='api.images')),
            ],
        ),
    ]