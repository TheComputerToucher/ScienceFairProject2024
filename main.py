import matplotlib.pyplot as pyplot
from mlstuff import *
from data import *
import json

DATA_FILE = "./data.csv"

time_steps = 12
epochs = 20

try:
    json_file = None
    with open("config.json", "r") as f:
        json_file = json.load(f)
    epochs = int(json_file["epochs"])
    DATA_FILE = json_file["data_path"]
except Exception as e:
    print(f"Failed to load config with error {e}, using defaults of DATA_FILE={DATA_FILE} and epochs={epochs}")

rnn = make_rnn(3, 1, (time_steps, 1), activation=['tanh', 'tanh'])

train_data, test_data, data = get_data(DATA_FILE)

train_model_on_data(rnn, train_data, time_steps, epochs)

print("Model successfully trained on CSV file ", DATA_FILE)

rnn.save_weights("./model.keras")

trainX, trainY = get_XY_from_data(train_data, time_steps)
testX, testY = get_XY_from_data(test_data, time_steps)

train_prediction = rnn.predict(trainX)
test_prediction = rnn.predict(testX)

print_error(trainY, testY, train_prediction, test_prediction)
write_to_csv(trainY, testY, train_prediction, test_prediction)

# print(rnn.predict(np.reshape(np.array([1,2,3]),(1, 3, 1))))

pyplot.ion()

fig, line1, line2 = plot_results(trainY, testY, train_prediction, test_prediction)

pyplot.xlabel("X-axis")
pyplot.ylabel("Y-axis")
pyplot.title("Updating plot...")

while True:
    fig.canvas.draw()
    fig.canvas.flush_events()

