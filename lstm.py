import math
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
import csv
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class LstmRNN(nn.Module):
    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.forwardCalculation = nn.Linear(hidden_size, output_size)

    def forward(self, _x):
        x, _ = self.lstm(_x)
        s, b, h = x.shape
        x = x.view(s * b, h).to(device)
        x = self.forwardCalculation(x)
        x = x.view(s, b, -1).to(device)
        return x


if __name__ == "__main__":
    # create database

    filename = 'C:\\Users\\22097\\Desktop\\LBMA-GOLD.csv'
    with open(filename, 'r') as filecsv:
        csvreader = csv.reader(filecsv)
        # print(csvreader)
        data = []
        for row in csvreader:
            try:
                print(float(row[1]))
                data.append(row[1])
            except:
                continue
    #print(data)
    data_len = len(data)-5
    print(data_len)
    t = np.linspace(0, 12 * np.pi, data_len)
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    bbb = np.cos(t) + np.sin(t)

    dataset = np.zeros((data_len, 2))
    dataset[:, 0] = [i for i in range(data_len)]
    dataset[:, 1] = data[:data_len]
   # dataset[:, 2] = bbb
    dataset = dataset.astype("float32")
    print(dataset)

    # plot part of the original dataset
    plt.figure()
    plt.plot( dataset[: data_len, 0], dataset[:data_len, 1], label="sin(t)")

    # plt.plot([2.5, 2.5], [-1.3, 0.55], "r--", label="t = 2.5")  # t = 2.5
    # plt.plot([6.8, 6.8], [-1.3, 0.85], "m--", label="t = 6.8")  # t = 6.8
    # plt.xlabel("t")
    # plt.ylim(-1.2, 1.2)
    # plt.ylabel("sin(t) and cos(t)")
    # plt.legend(loc="upper right")

    # choose dataset for training and testing
    train_data_ratio = 0.5  # Choose 80% of the data for testing
    train_data_len = int(data_len * train_data_ratio)
    train_x = dataset[:train_data_len, 0]
    train_y = dataset[:train_data_len, 1]
#    train_b = dataset[:train_data_len, 2]
    INPUT_FEATURES_NUM = 1
    OUTPUT_FEATURES_NUM = 1
    t_for_training = dataset[:train_data_len, 0]

    # test_x = train_x
    # test_y = train_y
    test_x = dataset[train_data_len:, 0]
    test_y = dataset[train_data_len:, 1]
 #   test_b = dataset[train_data_len:, 2]
    t_for_testing = dataset[train_data_len:, 0]

    # ----------------- train -------------------
    train_x_tensor = train_x.reshape(-1, 5, INPUT_FEATURES_NUM)  # set batch size to 5
    train_y_tensor = train_y.reshape(-1, 5, OUTPUT_FEATURES_NUM)  # set batch size to 5

    # transfer data to pytorch tensor
    train_x_tensor = torch.from_numpy(train_x_tensor).to(device)
    train_y_tensor = torch.from_numpy(train_y_tensor).to(device)
    # test_x_tensor = torch.from_numpy(test_x)

    lstm_model = LstmRNN(
        INPUT_FEATURES_NUM, 16, output_size=OUTPUT_FEATURES_NUM, num_layers=3
    ).to(device)  # 16 hidden units
    print("LSTM model:", lstm_model)
    print("model.parameters:", lstm_model.parameters)

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-2)

    max_epochs = 1000000
    for epoch in range(max_epochs):
        output = lstm_model(train_x_tensor)
        loss = loss_function(output, train_y_tensor)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if loss.item() < 1e-6:
            print(
                "Epoch [{}/{}], Loss: {:.5f}".format(epoch + 1, max_epochs, loss.item())
            )
            print("The loss value is reached")
            break
        elif (epoch + 1) % 1000 == 0:
            print(
                "Epoch: [{}/{}], Loss:{:.5f}".format(epoch + 1, max_epochs, loss.item())
            )

    # prediction on training dataset
    predictive_y_for_training = lstm_model(train_x_tensor)
    predictive_y_for_training = predictive_y_for_training.view(
        -1, OUTPUT_FEATURES_NUM
    ).data.cpu().numpy()

    # torch.save(lstm_model.state_dict(), 'model_params.pkl') # save model parameters to files

    # ----------------- test -------------------
    # lstm_model.load_state_dict(torch.load('model_params.pkl'))  # load model parameters from files
    lstm_model = lstm_model.eval()  # switch to testing model

    # prediction on test dataset
    test_x_tensor = test_x.reshape(
        -1, 5, INPUT_FEATURES_NUM
    )  # set batch size to 5, the same value with the training set
    test_x_tensor = torch.from_numpy(test_x_tensor).to(device)

    predictive_y_for_testing = lstm_model(test_x_tensor)
    predictive_y_for_testing = predictive_y_for_testing.view(
        -1, OUTPUT_FEATURES_NUM
    ).data.cpu().numpy()

    # ----------------- plot -------------------
    plt.figure()
    # plt.plot(t_for_training, train_x, "g", label="sin_trn")
    plt.plot(t_for_training, train_y, "b", label="ref_cos_trn")
    plt.plot(t_for_training, predictive_y_for_training, "y--", label="pre_cos_trn")

    # plt.plot(t_for_testing, test_x, "c", label="sin_tst")
    plt.plot(t_for_testing, test_y, "k", label="ref_cos_tst")
    plt.plot(t_for_testing, predictive_y_for_testing, "m--", label="pre_cos_tst")

    # plt.plot(
    #     [t[train_data_len], t[train_data_len]],
    #     [-1.2, 4.0],
    #     "r--",
    #     label="separation line",
    # )  # separation line

    # plt.xlabel("t")
    # plt.ylabel("sin(t) and cos(t)")
    # plt.xlim(t[0], t[-1])
    # plt.ylim(-1.2, 4)
    # plt.legend(loc="upper right")
    # plt.text(14, 2, "train", size=15, alpha=1.0)
    # plt.text(20, 2, "test", size=15, alpha=1.0)

    plt.show()


# x = np.arange(100)* math.pi /25
# y = np.cos(x)
# y1=np.sin(x)
# print(x,y)
# fig1=plt.figure
# plt.plot(x,y)
# plt.plot(x,y1)
# plt.show()
