from flask import Flask,render_template,request
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sklearn


app = Flask(__name__)

df = pickle.load(open("data.pkl","rb"))
ohe = pickle.load(open("new_ohe.pkl","rb"))
model = pickle.load(open("new_model.pkl","rb"))

ss = StandardScaler()

online_order = df.online_order.sort_values().unique()
book_table = df.book_table.sort_values().unique()
votes = df.votes.sort_values().unique()
rest_type = df.rest_type.sort_values().unique()
cuisines = df.cuisines.sort_values().unique()
approx_cost = df["approx_cost(for two people)"].sort_values().unique()
listed_in = df["listed_in(type)"].sort_values().unique()
reviews = df.reviews.sort_values().unique()

@app.route('/', methods =["GET", "POST"])
def Home():
    if request.method=="POST":
        order = request.form.get('online_order')
        book_tabel = request.form.get('book_table')
        vot = request.form.get('votes')
        restype = request.form.get('rest_type')
        cusin = request.form.get('cuisines')
        cost = request.form.get('approx_cost')
        listin = request.form.get('listed_in')
        rev = request.form.get('reviews')

        vals = pd.DataFrame([order,book_tabel,vot,restype,cusin,cost,listin,rev]).T
        valu = ohe.transform(vals)
        pred = model.predict(valu)[0]
        pred = round(pred,2)

        return render_template("output.html",output=pred)
    return render_template("index.html",online_order=online_order,book_table=book_table,votes=votes,rest_type=rest_type,cuisines=cuisines,approx_cost=approx_cost,listed_in=listed_in,reviews=reviews)


app.run(debug=True)