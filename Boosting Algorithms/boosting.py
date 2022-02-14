"""Gradient Boosting algorithm."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

df = pd.read_csv("dataset/WineQT.csv")
df = df.drop(["Id"], axis=1)

X = df.drop(["quality"], axis=1)
y = df["quality"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, shuffle=True, stratify=y, random_state=25
)


def train_and_score_model(model) -> None:
    """X."""
    model.fit(X_train, y_train)
    print("Score: ", model.score(X_test, y_test))


gb_clf = GradientBoostingClassifier()
train_and_score_model(gb_clf)  # Score:  0.591304347826087

gb_clf = GradientBoostingClassifier(n_estimators=250)
train_and_score_model(gb_clf)  # Score:  0.6086956521739131

gb_clf = GradientBoostingClassifier(n_estimators=500)
train_and_score_model(gb_clf)  # Score:  0.6

rf_clf = RandomForestClassifier()
train_and_score_model(rf_clf)  # Score:  0.6521739130434783

rf_clf = RandomForestClassifier(n_estimators=250)
train_and_score_model(rf_clf)  # Score:  0.6521739130434783

rf_clf = RandomForestClassifier(n_estimators=500)
train_and_score_model(rf_clf)  # Score:  0.6434782608695652

rf_clf = RandomForestClassifier(max_depth=10)
train_and_score_model(rf_clf)  # Score:  0.6260869565217392

cb_clf = CatBoostClassifier(verbose=0)
train_and_score_model(cb_clf)  # Score:  0.591304347826087

cb_clf = CatBoostClassifier(n_estimators=250, verbose=0)
train_and_score_model(cb_clf)  # Score:  0.6

cb_clf = CatBoostClassifier(n_estimators=500, verbose=0)
train_and_score_model(cb_clf)  # Score:  0.6347826086956522

cb_clf = CatBoostClassifier(max_depth=15, verbose=0)
train_and_score_model(cb_clf)  # Score:  0.6260869565217392

xgb_clf = XGBClassifier(verbosity=0)
train_and_score_model(xgb_clf)

xgb_clf = XGBClassifier(n_estimators=100, verbosity=0)
train_and_score_model(xgb_clf)

xgb_clf = XGBClassifier(n_estimators=400, verbosity=0)
train_and_score_model(xgb_clf)

xgb_clf = XGBClassifier(n_estimators=800, verbosity=0)
train_and_score_model(xgb_clf)


svc = SVC()
train_and_score_model(svc)  # Score:  0.45217391304347826


gb_clf = GradientBoostingClassifier()
train_and_score_model(gb_clf)  # Score:  0.3361963190184049

gb_clf = GradientBoostingClassifier(n_estimators=500)
train_and_score_model(gb_clf)  # Score:  0.34662576687116564

gb_clf = GradientBoostingClassifier(n_estimators=1000)
train_and_score_model(gb_clf)  # Score:  0.35030674846625764

gb_clf = GradientBoostingClassifier(n_estimators=500, max_depth=5)
train_and_score_model(gb_clf)  # Score:  0.3693251533742331

gb_clf = GradientBoostingClassifier(n_estimators=500, max_depth=10)
train_and_score_model(gb_clf)  # Score:  0.3374233128834356

gb_clf = GradientBoostingClassifier(n_estimators=500, max_depth=20)
train_and_score_model(gb_clf)  # Score:  0.3239263803680982
