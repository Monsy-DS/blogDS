from flask import render_template,url_for,request, Blueprint, redirect
from dsproject.core.models import Funkce_NLP, Trump_speech, Basket_funkce, Akcie_return
from wtforms import StringField, IntegerField
from flask_wtf import FlaskForm
from wtforms import IntegerField, SubmitField
from wtforms.validators import DataRequired, Email, EqualTo
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, LSTM
import matplotlib.pyplot

core=Blueprint('core',__name__)

@core.route('/')
def index():
	return render_template('index.html')

@core.route('/usprezident', methods=['GET','POST'])
def usprezident():
	return render_template('usprezident.html')


@core.route('/trumpresult', methods=['GET','POST'])
def trumpresult():
	slovo = request.args.get('slovo')
	delka = request.args.get('delka')
	rikam = Trump_speech.trump_funkce(slovo, delka)
	return render_template('trumpresult.html', rikam=rikam)

@core.route('/NLP', methods=['GET','POST'])
def NLP():
	return render_template('NLP.html')

@core.route('/nlpresult')
def nlpresult():
	veta = request.args.get('veta')
	pocet_hvezd = Funkce_NLP.Vrati_hvezdu(veta)+1
	return render_template('nlpresult.html',  pocet_hvezd=pocet_hvezd)


@core.route('/basket', methods=['GET','POST'])
def basket():
	return render_template('basket.html')

@core.route('/basketresult', methods=['GET','POST'])	
def basketresult():
	nakup_slovo = request.args.get('nakup')
	nakupni_seznam = Basket_funkce.vrat_nakup(nakup_slovo)
	return render_template('basketresult.html', nakup_slovo=nakup_slovo, nakupni_seznam=nakupni_seznam)

@core.route('/aboutme')
def aboutme():
	return render_template('aboutme.html')
#new code here

@core.route('/akcie', methods=['GET','POST'])
def akcie():
	return render_template('akcie.html')

@core.route('/akcieresult', methods=['GET','POST'])
def akcieresult():
	akcie = request.args.get('akcie')
	result_stock = Akcie_return.akcie_funkce(akcie)
	return render_template('akcieresult.html', akcie=akcie, result_stock = result_stock)
