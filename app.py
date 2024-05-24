from flask import Flask, render_template, request, jsonify
import random

app = Flask(__name__)

BOT_MSGS = [
    "Hi, how are you?",
    "Ohh... I can't understand what you're trying to say. Sorry!",
    "I like to play games... But I don't know how to play!",
    "Sorry if my answers are not relevant. :))",
    "I feel sleepy! :("
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/bot', methods=['POST'])
def bot():
    user_msg = request.json.get('message')
    response = random.choice(BOT_MSGS)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
