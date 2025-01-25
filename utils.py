def validate_input(user_input):
    if not user_input or not isinstance(user_input, str):
        raise ValueError("Input must be a non-empty string")
    if len(user_input) > 1000:
        raise ValueError("Input too long")
    return user_input.strip()
