import pickle

model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

while True:

    news = input("Enter news text: ")

    data = vectorizer.transform([news])

    prediction = model.predict(data)

    print("Model Output:", prediction)

    if prediction[0] == 0:
        print("Prediction: Fake News\n")
    else:
        print("Prediction: Real News\n")