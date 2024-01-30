"""This is the main python file for the program, which loads the data, trains the RNN on the data, 
plots and prints the results and finally writes the weights to a file called model.keras"""
import json
from matplotlib import pyplot
from mlstuff import make_rnn, train_model_on_data, print_error, write_to_csv, plot_results
from data import get_xy_from_data, get_data

DATA_FILE = "./data.csv"

TIME_STEPS = 12
EPOCHS = 20

try:
    JSON_FILE = None
    with open("config.json", "r", encoding="utf-8") as f:
        JSON_FILE = json.load(f)
    EPOCHS = int(JSON_FILE["epochs"])
    DATA_FILE = JSON_FILE["data_path"]
except FileNotFoundError:
    print("Couldn't find config.json, using the default values, run config_tool.py to create it")
except json.decoder.JSONDecodeError:
    print("Couldn't load config.json, it's probably corrupted, please run config_tool.py")

rnn = make_rnn(3, 1, (TIME_STEPS, 1), activation=['tanh', 'tanh'])

train_data, test_data, data = get_data(DATA_FILE)

train_model_on_data(rnn, train_data, TIME_STEPS, EPOCHS)

print("Model successfully trained on CSV file ", DATA_FILE)

rnn.save_weights("./model.keras")

train_x, train_y = get_xy_from_data(train_data, TIME_STEPS)
test_x, test_y = get_xy_from_data(test_data, TIME_STEPS)

train_prediction = rnn.predict(train_x)
test_prediction = rnn.predict(test_x)

print_error(train_y, test_y, train_prediction, test_prediction)
write_to_csv(train_y, test_y, train_prediction, test_prediction)

# print(rnn.predict(np.reshape(np.array([1,2,3]),(1, 3, 1))))

pyplot.ion()

fig, line1, line2 = plot_results(train_y, test_y, train_prediction, test_prediction)

pyplot.xlabel("X-axis")
pyplot.ylabel("Y-axis")
pyplot.title(f"Results with {EPOCHS} EPOCHS")

while True:
    fig.canvas.draw()
    fig.canvas.flush_events()
