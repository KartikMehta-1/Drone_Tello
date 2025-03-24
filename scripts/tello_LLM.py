from djitellopy import Tello
import openai

# OpenAI API key setup
openai.api_key = "sk-your-openai-key"

def translate_command_to_sdk(user_command):
    """
    Translate natural language commands into Tello Python SDK functions using OpenAI's ChatGPT API.
    """
    prompt = f"""
    You are a Tello SDK expert. Translate the following natural language command into Tello Python SDK functions:
    Command: {user_command}
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert in the Tello drone Python SDK."},
                {"role": "user", "content": prompt},
            ],
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return None

def execute_command_on_tello(command):
    """
    Print or execute the translated SDK command for debugging purposes.
    """
    try:
        print(f"Translated SDK Command:\n{command}")
        # Uncomment below to execute with a real Tello drone
        # exec(command)
    except Exception as e:
        print(f"Error executing command: {e}")

# Main program
if __name__ == "__main__":
    user_input = input("Enter your command: ")
    translated_command = translate_command_to_sdk(user_input)
    if translated_command:
        execute_command_on_tello(translated_command)
    else:
        print("Failed to translate the command. Please try again.")
