import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Softmax, Dropout
from tensorflow.keras import Model
import numpy as np
import scipy
import os
from tqdm import tqdm
import shutil
from itertools import product


class MyModel(Model):
    def __init__(self, hidden_size, input_layer_dropout_rate=0, hidden_layer_dropout_rate=0):
        '''
        hidden_size: hidden size of the model
        num_cells: number of cells
        '''
        super(MyModel, self).__init__()
        self.hidden_size = hidden_size
        self.input_dropout = Dropout(input_layer_dropout_rate)
        self.hidden_layer = Dense(hidden_size, 'tanh', kernel_initializer='glorot_normal')
        self.hidden_dropout = Dropout(hidden_layer_dropout_rate)
        self.output_layer = Dense(1, None)

    @staticmethod
    def stable_softmax(z, mask=None, axis=-1):
        """
        Compute softmax values for each sets of scores in Z.
        each column of Z is a set of score.    

        z: output of last layer [num_times, num_cells, num_cells]
        mask: neighbor mask [num_cells, num_cells]
        output: [num_times, num_cells, num_cells]
        """
        if mask is None:
            mask = tf.ones_like(z)
        z = z - tf.reduce_max(z, axis=axis, keepdims=True)
        e_z = tf.math.exp(z)
        # mask = tf.tile(mask[tf.newaxis, ...], [e_z.shape[0], 1, 1])
        # e_z = tf.where(mask, 1e-8, e_z)
        e_z = mask * (e_z + 1e-8)

        output = tf.divide(e_z, tf.math.reduce_sum(e_z, axis=axis, keepdims=True))
        try:
            assert np.allclose(tf.math.reduce_sum(output, axis=-1), tf.ones([z.shape[0], z.shape[1]]))
        except:
            print(tf.math.reduce_sum(output, axis=-1))
            raise Exception("Stable Softmax not working correctly")
        return output
    
    def call(self, x, mask):
        # x shape: [num_times, num_cells, num_cells, 5]
        # mask: [num_cells, num_cells]
        # output shape: [num_times, num_cells, num_cells]
        output = self.input_dropout(x)
        output = self.hidden_layer(output)
        output = self.hidden_dropout(output)
        output = self.output_layer(output)
        output = tf.squeeze(output) # [num_times, num_cells, num_cells]

        output = self.stable_softmax(output, mask, axis=-1)
        return output


def loss_function(M_pred, model, mask, model_input, cell_population, lambda_coef):
    '''
    M_pred: predicted transition population [num_days, num_times, num_cells, num_cells]
    model_input: input of neural network [num_times, num_cells, num_cells, 5]
    mask shape: [num_cells, num_cells]
    cell_population: population at each cell
    '''

    def safe_log(vector, inverse_mask):
        # do not calculate logs for non-neighbor transition or when value <= 0
        inverse_mask = tf.math.logical_or(inverse_mask, vector <= 0)
        safe_vector = tf.where(
            inverse_mask, 
            1, vector)
        return tf.math.log(safe_vector)
        # return tf.where(
        #     inverse_mask, 
        #     0, tf.math.log(safe_vector))

    # we have to inverse mask here as we wish to mask out non-neighbor
    inverse_mask = tf.math.logical_not(mask)
    inverse_mask_M = tf.reshape(inverse_mask, [1, 1, inverse_mask.shape[0], inverse_mask.shape[1]])
    inverse_mask_M = tf.tile(inverse_mask_M, [M_pred.shape[0], M_pred.shape[1], 1, 1])
    
    # theta shape [num_times, num_cells, num_cells]
    theta = tf.cast(model(model_input, mask), dtype=tf.float64)
    # theta = tf.where(theta <= 0, 1e-8, theta)
    inverse_mask_theta = tf.reshape(inverse_mask, [1, inverse_mask.shape[0], inverse_mask.shape[1]])
    inverse_mask_theta = tf.tile(inverse_mask_theta, [theta.shape[0], 1, 1])
    # print("inverse_mask_theta", inverse_mask_theta.shape)
    
    # L_loss = tf.math.reduce_sum(M_pred * (1 - tf.math.log(M_pred) + tf.math.log(theta)))
    L_loss = M_pred * (1 - safe_log(M_pred, inverse_mask_M) + \
                       safe_log(theta, inverse_mask_theta))
    
    where_nan = tf.where(tf.math.is_nan(safe_log(theta, inverse_mask_theta)))
    for row in where_nan:
        time, from_cell, to_cell = row
        print("Before log", theta[time, from_cell, :])
        print("After log", safe_log(theta, inverse_mask_theta)[time, from_cell, :])
        raise

    L_loss = tf.math.reduce_sum(L_loss)

    constraint_loss_in = tf.math.reduce_sum(
        (cell_population[:, :-1] - tf.math.reduce_sum(M_pred, axis=-1)) ** 2
    ) 
    constraint_loss_out = tf.math.reduce_sum(
        (cell_population[:, 1:] - tf.math.reduce_sum(M_pred, axis=-2)) ** 2
    )
    constraint_loss = constraint_loss_in + constraint_loss_out

    total_loss = L_loss - lambda_coef * constraint_loss
    return -total_loss, L_loss, constraint_loss

def append_line_to_file(obj, fname):
    with open(fname, "a") as fp:
        fp.write(obj)

def calc_NAE(M_estimate, M_label, N):
    return tf.reduce_sum(tf.math.abs(M_estimate - M_label)) / tf.reduce_sum(N)
    # return np.sum(np.abs(M_estimate - M_label)) / np.sum(N)

def train(
    model, model_input, M_optimize, mask, cell_population, lambda_coef, 
    M_label, 
    learning_rate, input_dropout_rate, hidden_dropout_rate, 
    min_delta, patience, max_iter, 
    exp_folder
):
    def train_step():
        # loss and NAE calculated in this function is values BEFORE UPDATE
        # print("A", model.trainable_variables)
        theta = tf.cast(model(model_input, mask), dtype=tf.float64)
        theta_M = tf.multiply(theta, tf.expand_dims(cell_population[:, :-1], axis=-1))
        y_hat = np.sum(theta_M, axis=-1)
        E = np.sum((cell_population[:, 1:] - y_hat) ** 2)
        nae_theta_M = calc_NAE(
            theta_M, 
            M_label, cell_population[:, :-1])
        nae_optimize_M = calc_NAE(M_optimize, M_label, cell_population[:, :-1])        

        with tf.GradientTape() as tape:
            negative_total_loss, L_loss, constraint_loss = loss_function(
                M_optimize, model, mask, model_input, cell_population, lambda_coef
            )
            total_loss = -1 * negative_total_loss

        variables = model.trainable_variables + [M_optimize]
        # variables = model.trainable_variables
        # variables = [M_optimize]
        gradients = tape.gradient(negative_total_loss, variables)
        # gradients[-1] = gradients[-1] * MASK
        # print(gradients[:-1])

        optimizer.apply_gradients(zip(gradients, variables))
        # optimizer.apply_gradients(zip([gradients[-1]], [variables[-1]]))
        # print(np.sum([np.sum(np.abs(i)) for i in gradients[:-1]]))
        
        # print("Total loss", total_loss)
        # print("Gradient", tf.reduce_any(tf.math.is_nan(gradients[0])))
        # print("Loss", loss)
        return total_loss, L_loss, constraint_loss, nae_theta_M, nae_optimize_M, E

    fname = f'lr_{learning_rate}_lambda_{lambda_coef}_dropout_{input_dropout_rate}_{hidden_dropout_rate}.txt'
    total_loss_file = os.path.join(exp_folder, 'total_loss', fname)
    L_loss_file = os.path.join(exp_folder, 'L_loss', fname)
    constraint_loss_file = os.path.join(exp_folder, 'constraint_loss', fname)
    nae_theta_M_file = os.path.join(EXP_NAME, 'NAE_theta_M', fname)
    nae_optimize_M_file = os.path.join(EXP_NAME, 'NAE_optimized_M', fname)
    E_file = os.path.join(EXP_NAME, 'E', fname)
    theta_before_training_file = os.path.join(EXP_NAME, 'theta_before_training_file', fname[:-4])
    theta_after_training_file = os.path.join(EXP_NAME, 'theta_after_training_file', fname[:-4])

    optimizer = tf.keras.optimizers.Adam(learning_rate)
    best = -1000000000000000
    wait = 0

    # for var in model.trainable_variables:
    #     print(var)
    # exit()

    # save theta before training
    theta_before_training = tf.cast(model(model_input, mask), dtype=tf.float64)
    np.save(theta_before_training_file, theta_before_training)

    for _ in tqdm(range(max_iter)):
        # print("Epoch ", _)
        total_loss, L_loss, constraint_loss, nae_theta_M, nae_optimize_M, E = train_step()
        append_line_to_file(str(float(total_loss)) + '\n', total_loss_file)
        append_line_to_file(str(float(L_loss)) + '\n', L_loss_file)
        append_line_to_file(str(float(constraint_loss)) + '\n', constraint_loss_file)
        append_line_to_file(str(float(nae_theta_M)) + '\n', nae_theta_M_file)
        append_line_to_file(str(float(nae_optimize_M)) + '\n', nae_optimize_M_file)
        append_line_to_file(str(float(E)) + '\n', E_file)
        
    # save theta after training
    theta_after_training = tf.cast(model(model_input, mask), dtype=tf.float64)
    np.save(theta_after_training_file, theta_after_training)

    return E

def random_initialization_M(num_days, num_time_points, num_cells, cell_population, neighbor_mask):
    np.random.seed(1000)
    random_theta = np.random.rand(num_days, num_time_points - 1, num_cells, num_cells)
    M_init = random_theta * tf.expand_dims(cell_population[:, :-1], axis=-1)
    M_init = M_init * neighbor_mask
    return tf.Variable(M_init, trainable=True)

def constraint_satisfied_random_initialization_M(
    num_days, num_time_points, num_cells, cell_population, neighbor_mask
):
    def solve_lin_eqn(A_eq, b_eq):
        np.random.seed(1000)
        c = np.random.rand(num_cells * num_cells)
        res = scipy.optimize.linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None))
        return res.x

    M_init = np.zeros((num_days, num_time_points - 1, num_cells, num_cells))

    A_eq = np.zeros((2 * num_cells, num_cells * num_cells))
    b_eq = np.zeros(2 * num_cells)
    for i in range(num_cells):
        for j in range(num_cells):
            A_eq[i, i * num_cells + j] = neighbor_mask[i, j]
            A_eq[i + num_cells, i + num_cells * j] = neighbor_mask[i, j]

    for d in tqdm(range(num_days)):
        for t in range(num_time_points - 1):
            b_eq = np.concatenate((cell_population[d, t], cell_population[d, t+1]))
            res = solve_lin_eqn(A_eq, b_eq)
            M_init[d, t] = res.reshape(num_cells, num_cells)
            assert np.all(np.sum(M_init[d, t], axis=1) == cell_population[d, t])
            assert np.all(np.sum(M_init[d, t], axis=0) == cell_population[d, t+1])

    return tf.Variable(M_init, trainable=True)

# neighbor_mask = np.array([
#     [1, 1, 0],
#     [1, 1, 1], 
#     [0, 1, 1]
# ])
# constraint_satisfied_random_initialization_M(1, 1, 3, None, neighbor_mask)

def close_to_zero_initialization_M(num_days, num_time_points, num_cells, cell_population, neighbor_mask):
    M_init = 1e-5 * np.ones((num_days, num_time_points - 1, num_cells, num_cells))
    M_init = M_init * neighbor_mask
    return tf.Variable(M_init, trainable=True)

def staying_initialization_M(num_days, num_time_points, num_cells, cell_population, neighbor_mask):
    M_init = np.zeros((num_days, num_time_points-1, num_cells, num_cells))
    # M_init = M_init + 1e-5
    for d in range(num_days):
        for t in range(num_time_points-1):
            for c in range(num_cells):
                M_init[d,t,c,c] = cell_population[d,t,c]
    return tf.Variable(M_init, trainable=True, dtype=tf.float64)

# def grid_search(**kwargs):
#     keys = kwargs.keys()
#     for instance in product(*kwargs.values()):
#         yield dict(zip(keys, instance))
        
# def get_max_population(cell_population):
#     max_pop = np.max(cell_population)
#     return max_pop

# def normalize_data(data, max_pop):
#     return data / max_pop


if __name__ == '__main__':
    DATA_FOLDER = './processed_data/stay_data'

    # the last day has nothing bro :v
    U = np.load(os.path.join(DATA_FOLDER, 'nn_input.npy'))[:-1]
    N = np.load(os.path.join(DATA_FOLDER, 'location_population.npy'))[:-1, :, :-1]
    N = tf.convert_to_tensor(N)
    NUM_DAYS, NUM_TIME_POINTS, NUM_CELLS = N.shape
    M_LABELS = np.load(os.path.join(DATA_FOLDER, 'transition_population.npy'))[:-1, :, :-1, :-1]
    MASK = np.load(os.path.join(DATA_FOLDER, 'neighbor_matrix.npy'))[:-1, :-1]
    MASK = MASK.astype(np.bool_)
    
    # normalizing data
    # max_pop_N = get_max_population(N)
    # N = normalize_data(N, max_pop_N)
    # M_LABELS = normalize_data(M_LABELS, max_pop_N)

    # hyperparameters
    # EXP_NAME = f'result_iwata/exp_023_long_training_zero_init_stay_dataset'
    EXP_NAME = f'result_iwata/demo'
    # HYPER_PARAMS = {
    #     "learning_rate": [1e-4, 1e-3, 1e-2, 1e-1, 1],
    #     "lambda": [1, 10, 100, 1000, 10000]
    # }
    HYPER_PARAMS = {
        "learning_rate": [1e-2],
        "lambda": [1, 10, 100, 1000], 
        "input_dropout_rate": [0],
        "hidden_dropout_rate": [0]
    }

    # random_initialization_M, constraint_satisfied_random_initialization_M, close_to_zero_initialization_M, 
    # staying_initialization_M
    INIT_METHOD = staying_initialization_M
    HIDDEN_SIZE = 10
    PATIENCE = 10
    MIN_DELTA = 10000
    MIN_PERCENT_INCREASE = 0.0001
    MAX_ITER = 1000
    # MAX_ITER = 5

    # create folder
    if os.path.exists(EXP_NAME):
        for filename in os.listdir(EXP_NAME):
            file_path = os.path.join(EXP_NAME, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    else:
        os.mkdir(EXP_NAME)
    os.mkdir(os.path.join(EXP_NAME, 'total_loss'))
    os.mkdir(os.path.join(EXP_NAME, 'L_loss'))
    os.mkdir(os.path.join(EXP_NAME, 'constraint_loss'))
    os.mkdir(os.path.join(EXP_NAME, 'NAE_optimized_M'))
    os.mkdir(os.path.join(EXP_NAME, 'NAE_theta_M'))
    os.mkdir(os.path.join(EXP_NAME, 'E'))
    os.mkdir(os.path.join(EXP_NAME, 'theta_before_training_file'))
    os.mkdir(os.path.join(EXP_NAME, 'theta_after_training_file'))

    # N_ONE_DAY = tf.expand_dims(N[0], axis=0)
    # M_LABELS_ONE_DAY = tf.expand_dims(M_LABELS[0], axis=0)
    # M_stay = initialize_stay_M(NUM_DAYS, NUM_TIME_POINTS, NUM_CELLS, N)
    # print(calc_NAE(M_stay, M_LABELS, N))


    for lr in tqdm(
        HYPER_PARAMS['learning_rate'],
        desc="Grid search hyperparams", 
        total=len(HYPER_PARAMS['learning_rate']) * len(HYPER_PARAMS['lambda']) *\
              len(HYPER_PARAMS["input_dropout_rate"]) * len(HYPER_PARAMS["hidden_dropout_rate"])
    ):
        for ld in HYPER_PARAMS['lambda']:
            for input_dropout_rate in HYPER_PARAMS["input_dropout_rate"]:
                for hidden_dropout_rate in HYPER_PARAMS["hidden_dropout_rate"]:
                    # Create an instance of the model
                    # M_init = initialize_M(NUM_DAYS, NUM_TIME_POINTS, NUM_CELLS, tf.expand_dims(N[:, :-1], axis=-1), MASK)
                    # M_init.assign(normalize_data(M_init, max_pop_N))
                    # M_init = initialize_stay_M(NUM_DAYS, NUM_TIME_POINTS, NUM_CELLS, N)
                    M_init = INIT_METHOD(NUM_DAYS, NUM_TIME_POINTS, NUM_CELLS, N, MASK)

                    print("M init", M_init[0, 0, 0])
                    model = MyModel(HIDDEN_SIZE, input_dropout_rate, hidden_dropout_rate)
                    print(calc_NAE(M_init, M_LABELS, N[:, :-1]))

                    train(
                        model, U, M_init, MASK, N, ld, 
                        M_LABELS, 
                        lr, input_dropout_rate, hidden_dropout_rate, 
                        MIN_DELTA, PATIENCE, MAX_ITER,
                        EXP_NAME
                    )
                    # print(model.trainable_varia   bles) 
                # print(f"Minimum E error: {min_E}")
                # print(f"Best lambda for learning rate {lr}: {ld}")

            

