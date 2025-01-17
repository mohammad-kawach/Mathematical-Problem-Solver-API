from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from flask_cors import CORS
import sympy as sp
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
import numpy as np

app = Flask(__name__)
# Enable CORS for all routes
CORS(app)

# Load environment variables from the .env file
load_dotenv()

@app.route('/chat', methods=['POST'])
def chat():
    """
    Handle user input and return a response.
    """
    user_input = request.json.get('message')
    print(f"Received input: {user_input}")  # Debug log

    # Process the input and solve mathematical problems
    try:
        result, steps, graph_url = solve_math(user_input)
        response = {"response": f"The solution is: {result}"}
        if steps:
            response["steps"] = steps
        if graph_url:
            response["graph"] = graph_url
        return jsonify(response)
    except ValueError as e:
        return jsonify({"response": str(e)})
    except Exception as e:
        return jsonify({"response": "Sorry, I couldn't solve that. Please make sure your input is a valid mathematical problem."})

def solve_math(user_input):
    """
    Solve advanced mathematical problems using SymPy.
    """
    # Log the raw input for debugging
    print(f"Raw user input: {user_input}")

    # Replace '^' with '**' for proper exponentiation
    user_input = user_input.replace('^', '**')
    
    # Add explicit multiplication for terms like 2x
    import re
    user_input = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', user_input)

    # Define the symbol(s) used in the equation
    x, y, z = sp.symbols('x y z')  # Add more symbols as needed

    # Check if the input contains an equality (e.g., x + 2 = 5)
    if '=' in user_input:
        try:
            # Split the input into left-hand side (LHS) and right-hand side (RHS)
            lhs, rhs = user_input.split('=')
            # Convert both sides to SymPy expressions
            lhs = sp.sympify(lhs.strip())
            rhs = sp.sympify(rhs.strip())
            print(f"LHS: {lhs}, RHS: {rhs}")  # Log LHS and RHS
            # Solve the equation
            solutions = sp.solve(sp.Eq(lhs, rhs))
            steps = generate_steps(lhs, rhs, solutions)
            graph_url = generate_graph(lhs - rhs)
            return solutions, steps, graph_url
        except Exception as e:
            print(f"Error solving equation: {e}")  # Log the specific error
            raise ValueError("Input is not a valid mathematical expression or equation.")

    # If it's not an equation, try to parse the input as a SymPy expression
    try:
        parsed_input = sp.sympify(user_input)
        print(f"Parsed input: {parsed_input}")
        simplified = sp.simplify(parsed_input)
        graph_url = generate_graph(parsed_input)
        return simplified, None, graph_url
    except (sp.SympifyError, TypeError) as e:
        print(f"SymPy parsing error: {e}")  # Log the specific error
        raise ValueError("Input is not a valid mathematical expression or equation.")

def generate_steps(lhs, rhs, solutions):
    """
    Generate step-by-step solutions for an equation.
    """
    steps = []
    steps.append(f"Original equation: {lhs} = {rhs}")
    steps.append(f"Rearrange to: {lhs - rhs} = 0")
    steps.append(f"Solutions: {solutions}")
    return steps

def generate_graph(expression):
    """
    Generate a graph for the given expression and return the image URL.
    """
    try:
        # Create a plot
        x = sp.symbols('x')
        f = sp.lambdify(x, expression, 'numpy')

        # Generate x and y values
        import numpy as np
        x_vals = np.linspace(-10, 10, 400)
        y_vals = f(x_vals)

        # Plot the graph
        plt.figure(figsize=(6, 4))
        plt.plot(x_vals, y_vals, label=str(expression))
        plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
        plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
        plt.title(f"Graph of {expression}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid()

        # Save the plot to a BytesIO object
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        graph_url = f"data:image/png;base64,{base64.b64encode(img.getvalue()).decode()}"
        plt.close()
        return graph_url
    except Exception as e:
        return None

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)