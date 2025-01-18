# Mathematical Problem Solver API

A Flask-based REST API that solves mathematical equations, generates step-by-step solutions, and creates visual graphs of mathematical expressions.

## Features

- Solves algebraic equations and expressions
- Provides step-by-step solutions
- Generates visual graphs of mathematical functions
- Handles implicit multiplication (e.g., "2x" is interpreted as "2*x")
- Supports multiple mathematical operations including:
  - Basic arithmetic
  - Algebraic equations
  - Function plotting
  - Expression simplification

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:

git clone https://github.com/mohammad-kawach/Mathematical-Problem-Solver-API

cd Mathematical-Problem-Solver-API


2. Install the required dependencies:

pip install -r requirements.txt


Create a .env file in the root directory (optional):

# Add any environment variables if needed

Running the Application
Start the server by running:
python app.py

The server will start on http://localhost:5000
API Endpoints
POST /chat
Solves mathematical problems and returns solutions with graphs.
Request Body:
{
    "message": "2x + 3 = 7"
}

Response:
{
    "response": "The solution is: [2]",
    "steps": [
        "Original equation: 2*x + 3 = 7",
        "Rearrange to: 2*x - 4 = 0",
        "Solutions: [2]"
    ],
    "graph": "data:image/png;base64,..."
}

Dependencies

Flask: Web framework
SymPy: Mathematical computations
Matplotlib: Graph generation
NumPy: Numerical computations
Flask-CORS: Cross-Origin Resource Sharing
python-dotenv: Environment variable management
waitress: Production WSGI server

Error Handling
The API includes error handling for:

Invalid mathematical expressions
Syntax errors
Unsolvable equations
Graph generation failures

Usage Examples

Solve a simple equation:

curl -X POST http://localhost:5000/chat \
-H "Content-Type: application/json" \
-d '{"message": "2x + 3 = 7"}'


Simplify an expression:

curl -X POST http://localhost:5000/chat \
-H "Content-Type: application/json" \
-d '{"message": "x^2 + 2x + 1"}'

Development
The application uses:

waitress for production server
matplotlib in non-interactive mode for graph generation
CORS enabled for cross-origin requests
JSON responses for API communication

Contributing

Fork the repository
Create a feature branch
Commit your changes
Push to the branch
Create a Pull Request

License
[Add your license information here]
Contact
[Add your contact information here]

This README.md provides:
1. A clear overview of what the project does
2. Installation instructions
3. Usage examples
4. API documentation
5. Development information
6. Dependencies list
7. Error handling information
8. Contributing guidelines

You can customize it further by:
1. Adding your specific license information
2. Including your contact details
3. Adding more specific examples
4. Including deployment instructions
5. Adding screenshots or examples of the graphs generated
6. Including any specific configuration requirements

