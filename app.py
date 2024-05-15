from flask import Flask, request

app = Flask(__name__)
app.config["DEBUG"] = True

# Import your classify_tweet function from hello.py
from hello import classify_tweet

@app.route("/", methods=["GET", "POST"])
def hello_world():
    if request.method == "POST":
        tweet = request.form["tweet"]
        sentiment = classify_tweet(tweet)
        return f"<h1>Sentiment Analysis Result:</h1><p>The sentiment of the tweet is: {sentiment}</p>"

    return '''
        <html>
        <body>
            <h1>Tweet Sentiment Analysis</h1>
            <form method="post">
                <p><input name="tweet" /></p>
                <p><input type="submit" value="Submit" /></p>
            </form>
        </body>
        </html>
        '''
