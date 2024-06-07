from flask import Flask, request, jsonify
from flask_cors import CORS
from model_kmeans import KMeansModel
from model_logistic import LogisticModel

app = Flask(__name__)
CORS(app)

kmeans_model = KMeansModel('modele.pkl')
logistic_model = LogisticModel('model.pkl', {
    "Humidite": [13, 15],
    "Aw": [0.7, 0.9],
    "Proteine": [16, 18],
    "Amidon": [38, 42],
    "Fibre": [13, 17],
    "Calcium": [25, 30]
})

@app.route('/predict/kmeans', methods=['POST'])
def predict_kmeans():
    data = request.get_json()
    if 'Product' not in data or 'Proteine' not in data or 'Amidon' not in data or 'Calcium' not in data:
        return jsonify({'error': 'Product, Proteine, Amidon, and Calcium are required'}), 400

    product = data['Product']
    proteine = float(data['Proteine'])
    amidon = float(data['Amidon'])
    calcium = float(data['Calcium'])
    prediction = kmeans_model.predict(proteine, amidon, calcium)
    
    # Ajouter les résultats de chaque cluster à la réponse JSON
    cluster_proteine = prediction['Cluster_Proteine']
    cluster_amidon = prediction['Cluster_Amidon']
    cluster_calcium = prediction['Cluster_Calcium']
    
    return jsonify({
        'product': product,
        'prediction': prediction,
        'Cluster_Proteine': cluster_proteine,
        'Cluster_Amidon': cluster_amidon,
        'Cluster_Calcium': cluster_calcium
    })

@app.route('/predict/logistic', methods=['POST'])
def predict_logistic():
    data = request.get_json()
    if 'Product' not in data or 'Humidite' not in data or 'Aw' not in data or 'Proteine' not in data or 'Amidon' not in data or 'Fibre' not in data or 'Calcium' not in data:
        return jsonify({'error': 'Product, Humidite, Aw, Proteine, Amidon, Fibre, and Calcium are required'}), 400

    product = data['Product']
    input_data = {
        "Humidite": float(data['Humidite']),
        "Aw": float(data['Aw']),
        "Proteine": float(data['Proteine']),
        "Amidon": float(data['Amidon']),
        "Fibre": float(data['Fibre']),
        "Calcium": float(data['Calcium'])
    }
    compliance_result = logistic_model.predict(input_data)
    return jsonify({"product": product, "compliance": compliance_result})

if __name__ == "__main__":
    app.run(debug=True)
