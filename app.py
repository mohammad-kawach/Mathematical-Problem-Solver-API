from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from flask_cors import CORS
import sympy as sp
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# New imports
from scipy import stats
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)
CORS(app)
load_dotenv()


@app.route('/chat', methods=['POST'])
def chat():
    if not request.json or 'message' not in request.json:
        return jsonify({"error": "Missing message in request"}), 400

    user_input = request.json.get('message')
    print(f"Received input: {user_input}")

    try:
        result, steps, graph_url = solve_math(user_input)
        response = {"response": f"The solution is: {result}"}
        if steps:
            response["steps"] = steps
        if graph_url:
            response["graph"] = graph_url
        return jsonify(response)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return jsonify({
            "error": "Sorry, I couldn't solve that. Please make sure your input is a valid mathematical problem."
        }), 500


def handle_integration(expression, variable='x'):
    """Handle integration of expressions"""
    x = sp.Symbol(variable)
    try:
        # Convert string to SymPy expression
        expression = sp.sympify(expression)
        result = sp.integrate(expression, x)
        steps = [
            f"Original expression: {expression}",
            f"Integrating with respect to {variable}",
            f"Result: {result}",
            f"To verify, differentiate: {sp.diff(result, x)}"
        ]
        return result, steps
    except Exception as e:
        raise ValueError(f"Integration error: {str(e)}")


def handle_differentiation(expression, variable='x'):
    """Handle differentiation of expressions"""
    x = sp.Symbol(variable)
    try:
        result = sp.diff(expression, x)
        steps = [
            f"Original expression: {expression}",
            f"Differentiating with respect to {variable}",
            f"Result: {result}"
        ]
        return result, steps
    except Exception as e:
        raise ValueError(f"Differentiation error: {str(e)}")


"""
def handle_matrix_operations(matrix_input):
    # Handle matrix operations
    try:
        # Convert string input to numpy array first
        matrix_data = np.array(eval(matrix_input))

        # Convert to SymPy matrix
        matrix = sp.Matrix(matrix_data.tolist())

        results = {
            "Original Matrix": matrix_data.tolist(),
            "Shape": matrix_data.shape,
            "Determinant": float(matrix.det()) if matrix_data.shape[0] == matrix_data.shape[1] else None,
            "Transpose": matrix_data.T.tolist(),
        }

        # Add inverse if matrix is square
        if matrix_data.shape[0] == matrix_data.shape[1]:
            try:
                results["Inverse"] = np.linalg.inv(matrix_data).tolist()
                results["Eigenvalues"] = np.linalg.eigvals(matrix_data).tolist()
            except np.linalg.LinAlgError:
                results["Inverse"] = None
                results["Eigenvalues"] = None

        steps = [f"{key}: {value}" for key, value in results.items()]
        return results, steps
    except Exception as e:
        raise ValueError(f"Matrix operation error: {str(e)}")
"""

"""
def handle_statistics(data_input):
    #Handle statistical calculations
    try:
        # Convert string input to list of numbers
        data = np.array([float(x) for x in data_input.replace('[', '').replace(']', '').split(',')])

        results = {
            "Count": len(data),
            "Mean": float(np.mean(data)),
            "Median": float(np.median(data)),
            "Mode": float(stats.mode(data)[0]),
            "Standard Deviation": float(np.std(data)),
            "Variance": float(np.var(data)),
            "Min": float(np.min(data)),
            "Max": float(np.max(data)),
            "Range": float(np.ptp(data)),
            "Q1": float(np.percentile(data, 25)),
            "Q3": float(np.percentile(data, 75))
        }

        steps = [f"{key}: {value}" for key, value in results.items()]
        return results, steps
    except Exception as e:
        raise ValueError(f"Statistical calculation error: {str(e)}")
"""


def parse_natural_language(text):
    """Parse natural language math problems"""
    # Tokenize and clean the input
    text = text.lower()

    # Keywords mapping
    operations = {
        'integrate': 'integration',
        'derivative': 'differentiation',
        'differentiate': 'differentiation',
        'matrix': 'matrix',
        'statistics': 'statistics',
        'solve': 'equation'
    }

    # Detect operation type
    operation_type = None
    for key, value in operations.items():
        if key in text:
            operation_type = value
            break

    # Extract mathematical expression
    try:
        if 'of' in text:
            expression = text.split('of')[-1].strip()
        else:
            expression = text.split('integrate')[-1].strip()  # Handle "integrate x^2"
        return operation_type, expression
    except Exception as e:
        raise ValueError(f"Parsing error: {str(e)}")


def validate_input(user_input):
    """Validate user input"""
    if not user_input or not isinstance(user_input, str):
        raise ValueError("Input must be a non-empty string")
    if len(user_input) > 1000:
        raise ValueError("Input too long")
    return user_input.strip()


# Modify your solve_math function to include these new capabilities
def solve_math(user_input):
    user_input = validate_input(user_input)
    print(f"Raw user input: {user_input}")

    # Check for matrix operations
    """
    if user_input.strip().startswith('matrix'):
        try:
            matrix_input = user_input.replace('matrix', '').strip()
            result, steps = handle_matrix_operations(matrix_input)
            return result, steps, None
        except Exception as e:
            print(f"Matrix error: {e}")
            raise ValueError("Invalid matrix input")
    """

    """
    # Check for statistics operations
    if user_input.strip().startswith('statistics'):
        try:
            data_input = user_input.replace('statistics for', '').strip()
            result, steps = handle_statistics(data_input)
            return result, steps, None
        except Exception as e:
            print(f"Statistics error: {e}")
            raise ValueError("Invalid statistics input")
        """
    # Split multiple equations
    equations = [eq.strip() for eq in user_input.split(',')]

    # Split multiple equations
    equations = [eq.strip() for eq in user_input.split(',')]

    # Replace '^' with '**' for proper exponentiation
    equations = [eq.replace('^', '**') for eq in equations]

    # Add explicit multiplication for terms like 2x
    import re
    equations = [re.sub(r'(\d)([a-zA-Z])', r'\1*\2', eq) for eq in equations]

    # Define symbols
    x, y, z = sp.symbols('x y z')

    if any('=' in eq for eq in equations):
        try:
            system_equations = []
            for eq in equations:
                if '=' in eq:
                    lhs, rhs = eq.split('=')
                    lhs = sp.sympify(lhs.strip())
                    rhs = sp.sympify(rhs.strip())
                    system_equations.append(sp.Eq(lhs, rhs))
                else:
                    # Handle expressions without equals sign
                    system_equations.append(sp.sympify(eq))

            print(f"System equations: {system_equations}")

            # Solve the system of equations
            solutions = sp.solve(system_equations)
            steps = generate_steps_system(system_equations, solutions)
            graph_url = generate_graph_system(system_equations)
            return solutions, steps, graph_url
        except Exception as e:
            print(f"Error solving system: {e}")
            raise ValueError("Invalid system of equations.")

    # Handle single expression
    try:
        parsed_input = sp.sympify(user_input)
        print(f"Parsed input: {parsed_input}")
        simplified = sp.simplify(parsed_input)
        graph_url = generate_graph(parsed_input)
        return simplified, None, graph_url
    except (sp.SympifyError, TypeError) as e:
        print(f"SymPy parsing error: {e}")
        raise ValueError("Invalid mathematical expression.")


def generate_steps_system(equations, solutions):
    steps = []
    steps.append(f"Original system of equations:")
    for eq in equations:
        steps.append(f"  {eq}")
    steps.append(f"Solutions: {solutions}")
    return steps


def generate_graph(expression):
    """
    Generate a graph for a single expression.
    """
    try:
        # Create a plot
        x = sp.symbols('x')
        f = sp.lambdify(x, expression, 'numpy')

        # Generate x and y values
        x_vals = np.linspace(-10, 10, 400)
        y_vals = f(x_vals)

        # Filter out complex values and infinities
        mask = np.isfinite(y_vals) & np.isreal(y_vals)

        plt.figure(figsize=(6, 4))
        plt.plot(x_vals[mask], y_vals[mask], label=str(expression))
        plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
        plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
        plt.title(f"Graph of {expression}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid()

        # Set reasonable axis limits
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)

        # Save the plot to a BytesIO object
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        graph_url = f"data:image/png;base64,{base64.b64encode(img.getvalue()).decode()}"
        plt.close()
        return graph_url
    except Exception as e:
        print(f"Error generating graph: {e}")
        return None


def generate_graph_system(equations):
    try:
        # Check if we have equations in two variables (x and y)
        x, y = sp.symbols('x y')

        # Create a plot
        plt.figure(figsize=(8, 6))

        # Generate points for each equation
        x_vals = np.linspace(-10, 10, 400)

        for eq in equations:
            try:
                # Solve equation for y in terms of x
                y_expr = sp.solve(eq, y)[0]
                f = sp.lambdify(x, y_expr, 'numpy')
                y_vals = f(x_vals)

                # Filter out complex values and infinities
                mask = np.isfinite(y_vals) & np.isreal(y_vals)
                plt.plot(x_vals[mask], y_vals[mask], label=str(eq))
            except:
                continue

        plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
        plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
        plt.title("System of Equations")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid()

        # Set reasonable axis limits
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)

        # Save the plot to a BytesIO object
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        graph_url = f"data:image/png;base64,{base64.b64encode(img.getvalue()).decode()}"
        plt.close()
        return graph_url
    except Exception as e:
        print(f"Error generating graph: {e}")
        return None


if __name__ == "__main__":
    from waitress import serve

    serve(app, host="0.0.0.0", port=5000)
