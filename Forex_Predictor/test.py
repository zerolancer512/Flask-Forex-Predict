from xgboost import XGBClassifier, Booster
model = XGBClassifier()
booster = Booster()
booster.load_model('model.pkl')
model._Booster = booster

print(model.predict([[0,0,0,0,0,00]]))