from flask import Flask,request, jsonify, render_template_string
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Sample data
data = pd.DataFrame({
    'Movie1': [5, 4, 0, 0, 2],
    'Movie2': [3, 0, 0, 4, 1],
    'Movie3': [0, 2, 4, 0, 0],
    'Movie4': [0, 0, 0, 5, 3],
    'Movie5': [1, 5, 0, 0, 4]
}, index=['User1', 'User2', 'User3', 'User4', 'User5'])

# Preprocessing
data_filled = data.fillna(0)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_filled)
model = NearestNeighbors(n_neighbors=2, metric='cosine')
model.fit(data_scaled)

# HTML Template
html_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Movie Recommendation System</title>
</head>
<body>
    <h1>Movie Recommendation System</h1>
    <label for="user">Select User:</label>
    <select id="user">
        <option value="0">User1</option>
        <option value="1">User2</option>
        <option value="2">User3</option>
        <option value="3">User4</option>
        <option value="4">User5</option>
    </select>
    <button onclick="getRecommendations()">Get Recommendations</button>
    <div id="recommendations"></div>

    <script>
        function getRecommendations() {
            const userIndex = document.getElementById('user').value;
            fetch(`/recommendations?user_index=${userIndex}`)
                .then(response => response.json())
                .then(data => {
                    document.getElementById('recommendations').innerHTML = `
                        <h2>Recommended Movies:</h2>
                        <ul>
                            ${data.recommendations.map(movie => `<li>${movie}</li>`).join('')}
                        </ul>
                    `;
                });
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(html_template)

@app.route('/recommendations')
def recommendations():
    user_index = int(request.args.get('user_index'))
    distances, indices = model.kneighbors([data_scaled[user_index]])
    recommended_movies = []

    for i in indices[0]:
        if i != user_index:
            recommended_movies.extend(data.iloc[i].index[data.iloc[i] > 0])

    user_rated_movies = data.iloc[user_index][data.iloc[user_index] > 0].index
    recommendations = [movie for movie in set(recommended_movies) if movie not in user_rated_movies]

    return jsonify({'recommendations': recommendations})

if __name__ == '__main__':
    app.run(debug=True)
