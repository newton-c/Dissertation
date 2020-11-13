from classifier import *

clf = SentimentClassifier()

ex_1 = "Las FARC estaban ahi" 
ex_2 = "Las personas estaban ahi"
print(clf.predict(ex_1))
print(clf.predict(ex_2))

ex_3 = "Los guerrilleros atacaran al pueblo, matando a mas de un mil de  " \
       "personas inocentes"
ex_4 = "Los heroes ayudaron al pueblo reconstuir las casas"
ex_5 = "Las personas hicieron una cosa en un lugar"
print(clf.predict(ex_3))
print(clf.predict(ex_4))
print(clf.predict(ex_5))
