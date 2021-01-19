from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, TextAreaField, IntegerField
from wtforms.validators import DataRequired
from wtforms import validators

class USPREZ(FlaskForm):
	slovo = StringField('First_word', validators=[DataRequired()])
	hodnota = IntegerField('Temperature')
	pocet_znaku = IntegerField('Znaky')
	submit = SubmitField('Lets talk')

class NLP(FlaskForm):
	hodnoceni = TextAreaField('english_text')
	submit = SubmitField('number of stars')
