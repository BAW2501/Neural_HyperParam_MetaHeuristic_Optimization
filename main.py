from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from mealpy.evolutionary_based import FPA
def decode_solution(solution, data):
    # batch_size = 2**int(solution[0])
    # # 1 -> 1.99 ==> 1
    # # 2 -> 2.99 ==> 2
    # # 3 -> 3.99 ==> 3
    #
    # epoch = 10 * int(solution[1])
    # # 10 * 70 = 700
    # # 10 * 200 = 2000
    #
    # opt_integer = int(solution[2])
    # opt = OPT_ENCODER.inverse_transform([opt_integer])[0]
    # # 0 - 0.99 ==> 0 index ==> should be SGD (for example)
    # # 1 - 1.99 ==> 1 index ==> should be RMSProp
    #
    # learning_rate = solution[3]
    #
    # network_weight_initial_integer = int(solution[4])
    # network_weight_initial = WOI_ENCODER.inverse_transform([network_weight_initial_integer])[0]
    #
    # act_integer = int(solution[5])
    # activation = ACT_ENCODER.inverse_transform([act_integer])[0]
    #
    # n_hidden_units = int(solution[6])

    batch_size = 2 ** int(solution[0])
    epoch = 10 * int(solution[1])
    opt_integer = int(solution[2])
    opt = data["OPT_ENCODER"].inverse_transform([opt_integer])[0]
    learning_rate = solution[3]
    act_integer = int(solution[5])
    activation = data["ACT_ENCODER"].inverse_transform([act_integer])[0]
    n_hidden_units = int(solution[6])
    return {
        "batch_size": batch_size,
        "epoch": epoch,
        "opt": opt,
        "learning_rate": learning_rate,
        "activation": activation,
        "n_hidden_units": n_hidden_units,
    }
    
def generate_accuracy_value(structure, data):
    # structure is the return value of decode_solution
    # make model from MLPClassifier
    model = MLPClassifier(
        hidden_layer_sizes=(structure["n_hidden_units"],),
        activation=structure["activation"],
        solver=structure["opt"],
        alpha=structure["learning_rate"],
        max_iter=structure["epoch"],
        batch_size= structure["batch_size"],
        
    )
    model.fit(data["X_train"], data["y_train"])
    acc = model.score(data["X_test"], data["y_test"])
    return acc

def fitness_function(solution, data):
    # 1. decode solution
    # 2. generate accuracy value
    # 3. return accuracy value
    sol = decode_solution(solution, data)
    acc = generate_accuracy_value(sol, data)
    print(f"Accuracy: {acc}")
    return acc
    

if __name__ == '__main__':
 # LABEL ENCODER
    OPT_ENCODER = LabelEncoder()
    OPT_ENCODER.fit(['sgd', 'adam', 'lbfgs'])  # domain range ==> 7 values
    ACT_ENCODER = LabelEncoder()
    ACT_ENCODER.fit(['identity', 'logistic', 'tanh', 'relu'])
    
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    DATA = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        'OPT_ENCODER': OPT_ENCODER,
        'ACT_ENCODER': ACT_ENCODER,
    }

    LB = [1, 5, 0, 0.01, 0, 0, 5]
    UB = [3.99, 20.99, 6.99, 0.5, 7.99, 7.99, 50]

    problem = {
        "fit_func": lambda solution: fitness_function(solution, DATA),
        "lb": LB,
        "ub": UB,
        "minmax": "max",
        "log_to": None,
        "save_population": False,
        "data": DATA,
    }
    model = FPA.OriginalFPA(epoch=5, pop_size=20)
    # model = GWO.OriginalGWO(epoch=5, pop_size=20)
    model.solve(problem)

    print(f"Best solution: {model.solution[0]}")
    sol = decode_solution(model.solution[0], DATA)

    print(f"Batch-size: {sol['batch_size']}, Epoch: {sol['epoch']}, Opt: {sol['opt']}, "
          f"Learning-rate: {sol['learning_rate']}, NWI: {sol['network_weight_initial']}, "
          f"Activation: {sol['activation']}, n-hidden: {sol['n_hidden_units']}")