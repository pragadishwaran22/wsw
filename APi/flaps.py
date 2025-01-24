from flask import Flask, request
from flask_restful import Resource, Api
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB

app = Flask(__name__)
api = Api(app)
iris = load_iris()
X = iris.data
y = iris.target

clf = GaussianNB()
clf.fit(X, y)
class IrisClassification(Resource):
    def post(self):
        data = request.get_json()
        test_data = [[
            data['sepal_length'],
            data['sepal_width'],
            data['petal_length'],
            data['petal_width'],
        ]]
        class_idx = clf.predict(test_data)[0]
        return {'class': iris.target_names[class_idx]}

api.add_resource(IrisClassification, '/predict')
if __name__ == '__main__':
    app.run(debug=True)