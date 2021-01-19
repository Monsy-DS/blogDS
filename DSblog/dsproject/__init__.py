from flask import Flask
import os

app = Flask(__name__)
app.config['SECRET_KEY']='totally_secret_password_number_dve'
app.static_folder = 'static'
#####################################
###### Blueprints registration ######
#####################################

#from slozka.podslozka.file import nazev_blueprint
#app.register_blueprint(nazev_blueprintu)
from dsproject.core.views import core

app.register_blueprint(core)
