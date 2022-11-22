import pickle
import time

from inference import tag_all_test
from optimization import get_optimal_vector
from preprocessing import preprocess_train


def train_model(train_path, weights_path, threshold, lam, f200, f300):
    start = time.time()
    statistics, feature2id = preprocess_train(train_path, threshold, f200, f300)
    get_optimal_vector(
        statistics=statistics,
        feature2id=feature2id,
        weights_path=weights_path,
        lam=lam,
    )
    end = time.time()
    model1_time = end - start
    print(f"model train time: {model1_time}")


def test_model(weights_path, test_path, predictions_path):
    start = time.time()
    with open(weights_path, "rb") as f:
        optimal_params, feature2id = pickle.load(f)
    pre_trained_weights = optimal_params[0]

    print(pre_trained_weights)
    tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path)
    end = time.time()
    model1_time = end - start
    print(f"model test + evaluate time: {model1_time}")


def generate_comp(weights_path, test_path, predictions_path):
    start = time.time()
    with open(weights_path, "rb") as f:
        optimal_params, feature2id = pickle.load(f)
    pre_trained_weights = optimal_params[0]

    print(pre_trained_weights)
    tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path)
    end = time.time()
    model1_time = end - start
    print(f"model test + evaluate time: {model1_time}")


def main():
    # ---- Files Paths  -----------
    train_1_path = "data/train1.wtag"
    train_1_test_itself_input = "data/train1 test itself.wtag"
    train_1_test_itself_output = "data/train1 test.wtag"
    test_1_input_path = "data/test1.wtag"
    test_1_output_path = "data/test1_predictions.wtag"
    weights_1_path = "weights_1.pkl"

    train_2_path = "data/train2.wtag"
    weights_2_path = "weights_2.pkl"

    # -----------------------------
    # Train Models
    # -----------------------------

    # train_model(
    #     train_1_path,
    #     weights_1_path,
    #     threshold=3,
    #     lam=1,
    #     f200=True,
    #     f300=True,
    # )
    # train_model(train_2_path, weights_2_path, threshold=3, lam=0.5, f200=True, f300=True)

    # -----------------------------
    # Test Model 1
    # -----------------------------

    # test_model(weights_1_path, train_1_test_itself_input, train_1_test_itself_output)

    # -----------------------------
    # Generate Comp Tagged
    # -----------------------------
    # generate_comp(weights_1_path, comp_1_input_path, comp_1_output_path)
    # generate_comp(weights_2_path, comp_2_input_path, comp_2_output_path)


# def CV():
#     from itertools import product
#     threshold_l = [1, 3, 5]
#     lam_l = [1, 2, 0.5]
#     B_l = [5, 2]
#     for lam, B, threshold in product(lam_l, B_l, threshold_l):
#         print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
#         print(f"#######    {threshold=}, {lam=}, {B=}    #########")
#         accuracy_list = []
#         mean_f1_list = []
#         median_f1_list = []
#         # cross validation
#         for i in range(1, 6):
#             train_path = f"data_cv/cv_train_{i}.wtag"
#             test_input_path = f"data_cv/cv_test_{i}.wtag"
#             test_output_path = "test_predictions.wtag"
#             weights_path = "weights.pkl"
#             print("#######    train    ########")
#             train_model(train_path, weights_path, threshold=threshold, lam=lam, f200=True, f300=True)
#             print("#######    predict    ########")
#             accuracy, mean_f1, median_f1 = test_model(weights_path, test_input_path, test_output_path, B=B)
#             accuracy_list.append(accuracy)
#             mean_f1_list.append(mean_f1)
#             median_f1_list.append(median_f1)
#
#         cv_accuracy = sum(accuracy_list) / len(accuracy_list)
#         cv_mean_f1 = sum(mean_f1_list) / len(mean_f1_list)
#         cv_median_f1 = sum(median_f1_list) / len(median_f1_list)
#
#         with open("CV_res.txt", 'a') as f:
#             f.write(f"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
#             f.write(f"#######    {threshold=}, {lam=}, {B=}    #########\n")
#             f.write(f"{cv_accuracy=}, {cv_mean_f1=}, {cv_median_f1=}\n\n")


if __name__ == "__main__":
    main()
