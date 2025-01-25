from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from waitress import serve
import logging
from math_operations import solve_math
from utils import validate_input
from flask_cors import CORS

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})  # Replace with your frontend's URL

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@app.route('/chat', methods=['POST'])
def chat():
    if not request.json or 'message' not in request.json:
        return jsonify({"error": "Missing message in request"}), 400

    user_input = request.json.get('message')
    logging.info(f"Received input: {user_input}")

    try:
        user_input = validate_input(user_input)
        result, steps, graph_url = solve_math(user_input)
        response = {"response": f"The solution is: {result}"}
        if steps:
            response["steps"] = steps
        if graph_url:
            response["graph"] = graph_url
        return jsonify(response)
    except ValueError as e:
        logging.error(f"ValueError: {str(e)}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return jsonify({
            "error": "Sorry, I couldn't solve that. Please make sure your input is a valid mathematical problem."
        }), 500


if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=5000)