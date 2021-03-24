from utils import *
from models import *


def main():
    
    print("Training on dataset 0...")
    label = 0
    X_train, y_train = get_training_data(label=label, suffix="")
    sigma_dic = {3: 13, 4:7, 5:10, 6:7, 7:10, 8:7}

    kernel_list = []
    for k in [3, 4, 5, 6, 7, 8]:
        kernel_list.append({"name":"spectrum_kernel", "sub_kernel":"RBF",
                                                        "k": k, "sigma":sigma_dic[k]})


    K_train = sum_kernel(X_train, X_train, kernel_list)

    X_test = get_test_data(label=label, suffix="")


    K_test = sum_kernel(X_test, X_train, kernel_list)
    model = SVM(K=K_train, y_train=y_train, mu=1e-3)
    model.fit(loss="squared_hinge")
    pred0=model.predict(K_test)


    print("Training on dataset 1...")
    label = 1
    X_train, y_train = get_training_data(label=label, suffix="")
    sigma_dic = {3: 7, 4:10, 5:7, 6:7, 7:13, 8:10}

    kernel_list = []
    for k in [3, 4, 5, 6, 7, 8]:
        kernel_list.append({"name":"spectrum_kernel", "sub_kernel":"RBF",
                                                        "k": k, "sigma":sigma_dic[k]})


    K_train = sum_kernel(X_train, X_train, kernel_list)

    X_test = get_test_data(label=label, suffix="")


    K_test = sum_kernel(X_test, X_train, kernel_list)
    model = SVM(K=K_train, y_train=y_train, mu=1e-3)
    model.fit(loss="squared_hinge")
    pred1=model.predict(K_test)


    print("Training on dataset 2...")
    label = 2
    X_train, y_train = get_training_data(label=label, suffix="")
    #dic = dic = {3: 5, 4:7, 5:7, 6:7, 7:10, 8:13}
    sigma_dic = {3: 5, 4:7, 5:7, 6:7, 7:10, 8:13}
    kernel_list = []
    for k in [3, 4, 5, 6, 7, 8]:
        kernel_list.append({"name":"spectrum_kernel", "sub_kernel":"RBF",
                                                        "k": k, "sigma":sigma_dic[k]})


    K_train = sum_kernel(X_train, X_train, kernel_list)

    X_test = get_test_data(label=label, suffix="")


    K_test = sum_kernel(X_test, X_train, kernel_list)
    model = SVM(K=K_train, y_train=y_train, mu=1e-3)
    model.fit(loss="squared_hinge")
    pred2=model.predict(K_test)


    format_predictions([np.concatenate(pred0), np.concatenate(pred1), np.concatenate(pred2)], result_path="./Yte.csv")
    
if __name__ == '__main__':
    main()