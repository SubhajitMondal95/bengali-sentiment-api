from app.predictor import predict_sentiment

sentence = "আমি আজ খুব আনন্দিত"

result = predict_sentiment(sentence)

print(result)