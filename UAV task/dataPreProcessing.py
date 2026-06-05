import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

from functions import set_seed

def dataset_masking(dataset=None):

    # Add dt column at position 1 (limiting the number to 6 decimal points)
    dataset.insert(1, 'dt', dataset['time'].diff().fillna(0).round(6))  # Calculate time intervals (dt) between consecutive samples and place it in a new column 'dt' in position 1

    # Add Euler angles columns (roll, pitch, yaw) at positions 17, 18, 19
    dataset.insert(17, 'roll', 0.0)
    dataset.insert(18, 'pitch', 0.0)
    dataset.insert(19, 'yaw', 0.0)

    # Add the following non-used columns to match the old dataset format
    dataset.insert(24, 'pwm_1', 0.0)
    dataset.insert(25, 'pwm_2', 0.0)
    dataset.insert(26, 'pwm_3', 0.0)
    dataset.insert(27, 'pwm_4', 0.0)
    dataset.insert(28, 'total_thrust', 0.0)

    # Drop the unused columns (from 37 to the end)
    dataset = dataset.drop(columns=dataset.columns[37:])

    return dataset



def from_quaternion_to_euler(dataset=None):
    """
    Convert quaternion orientation to Euler angles (roll, pitch, yaw) in the dataset.
    Quaternion format in dataset: [q_w, q_x, q_y, q_z]
    Euler angles format: [roll, pitch, yaw]
    """
    q_w = dataset[:, 20:21]
    q_x = dataset[:, 21:22]
    q_y = dataset[:, 22:23]
    q_z = dataset[:, 23:24]

    # Compute roll (x-axis rotation)
    sinr_cosp = 2 * (q_w * q_x + q_y * q_z)
    cosr_cosp = 1 - 2 * (q_x**2 + q_y**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Compute pitch (y-axis rotation)
    sinp = 2 * (q_w * q_y - q_z * q_x)
    pitch = np.where(np.abs(sinp) >= 1, np.sign(sinp) * (np.pi / 2), np.arcsin(sinp))

    # Compute yaw (z-axis rotation)
    siny_cosp = 2 * (q_w * q_z + q_x * q_y)
    cosy_cosp = 1 - 2 * (q_y**2 + q_z**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    # Replace quaternion columns with Euler angles in the dataset (rounding to 6 decimal points)
    dataset[:, 17:18] = np.round(roll, 6)
    dataset[:, 18:19] = np.round(pitch, 6)
    dataset[:, 19:20] = np.round(yaw, 6)

    return dataset



def px4_pwm_to_thrust(dataset = None, mass=2.0, g=9.81):
    """
    1) I need to clamp the values of cmd_thrust between -1 and 0, because out of this interval the drone will read only -1 and 0, so it doesn't make sense to have values outside this range
    2) The cmd_thrust in the dataset is a PWM (or another signal), so i need to apply a conversion factor to get the actual thrust force. This factor is determined empirically to match the hover condition
    """
    cmd_thrust = dataset[:, 29:30] 
    cmd_thrust = np.clip(cmd_thrust, -1.0, 0.0)  # Clamp between -1 and 0

    thrust = cmd_thrust * mass * g / 0.72 #-0.72 # conversion factor to get thrust in Newtons
    dataset[:, 29:30] = np.round(thrust, 6)

    return dataset

def px4_angular_rate_to_torque(dataset=None, inertia=np.array([0.0216, 0.0216, 0.04])):
    """
    Compute the actual torques applied to the drone by inverting the rotational dynamics equations.
    
    From Euler's rotational equations:
        I_x * alpha_x = (I_y - I_z) * wy * wz + tau_x
        I_y * alpha_y = (I_z - I_x) * wx * wz + tau_y
        I_z * alpha_z = (I_x - I_y) * wx * wy + tau_z
    
    Solving for torques:
        tau_x = I_x * alpha_x - (I_y - I_z) * wy * wz
        tau_y = I_y * alpha_y - (I_z - I_x) * wx * wz
        tau_z = I_z * alpha_z - (I_x - I_y) * wx * wy
    """
    I_x, I_y, I_z = inertia[0], inertia[1], inertia[2]

    # Angular velocities from the dataset
    w_roll = dataset[:, 11:12]
    w_pitch = dataset[:, 12:13]
    w_yaw = dataset[:, 13:14]

    # Angular accelerations from the dataset
    alpha_roll = dataset[:, 14:15]
    alpha_pitch = dataset[:, 15:16]
    alpha_yaw = dataset[:, 16:17]

    # Invert the rotational dynamics to get torques
    tau_x = I_x * alpha_roll - (I_y - I_z) * w_pitch * w_yaw
    tau_y = I_y * alpha_pitch - (I_z - I_x) * w_roll * w_yaw
    tau_z = I_z * alpha_yaw - (I_x - I_y) * w_roll * w_pitch

    # Store torques in the control columns
    dataset[:, 30:31] = np.round(tau_x, 6)
    dataset[:, 31:32] = np.round(tau_y, 6)
    dataset[:, 32:33] = np.round(tau_z, 6)

    return dataset


def get_mean_and_std(dataset=None):
    """
    Get mean and standard deviation of the whole dataset for normalization.
    """
    time = dataset[:, 0:1]
    dt = dataset[:, 1:2]
    linear_pos = dataset[:, 2:5] # x, y, z 
    linear_vel = dataset[:, 5:8] # vx, vy, vz
    linear_acc = dataset[:, 8:11] # a_x, a_y, a_z
    angular_vel = dataset[:, 11:14] # w_x, w_y, w_z
    angular_acc = dataset[:, 14:17] # alpha_x, alpha_y, alpha_z
    angular_pos = dataset[:, 17:20] # roll, pitch, yaw
    rest_of_the_data = dataset[:, 20:29] # other data not used for training
    controls = dataset[:, 29:33] # thrust, torque_roll, torque_pitch, torque_yaw

    lin_pos_mean = linear_pos.mean(axis=0)
    lin_pos_std = linear_pos.std(axis=0) + 1e-8
    
    lin_vel_mean = linear_vel.mean(axis=0)
    lin_vel_std = linear_vel.std(axis=0) + 1e-8
    
    lin_acc_mean = linear_acc.mean(axis=0)
    lin_acc_std = linear_acc.std(axis=0) + 1e-8
    
    ang_vel_mean = angular_vel.mean(axis=0)
    ang_vel_std = angular_vel.std(axis=0) + 1e-8
    
    ang_acc_mean = angular_acc.mean(axis=0)
    ang_acc_std = angular_acc.std(axis=0) + 1e-8
    
    # rest_mean = rest_of_the_data.mean(axis=0)
    # rest_std = rest_of_the_data.std(axis=0) + 1e-8
    
    controls_mean = controls.mean(axis=0)
    controls_std = controls.std(axis=0) + 1e-8

    mean = np.hstack((lin_pos_mean, lin_vel_mean, ang_vel_mean, lin_acc_mean, ang_acc_mean, controls_mean))
    std = np.hstack((lin_pos_std, lin_vel_std, ang_vel_std, lin_acc_std, ang_acc_std, controls_std))

    return mean, std


def configure_data(dataset):
    m, n = dataset.shape
    dt_test = dataset[:, 1:2]
    linear_pos = dataset[:,2:5]
    linear_vel = dataset[:,5:8]
    linear_acc = dataset[:, 8:11]
    angular_vel = dataset[:,11:14]
    angular_acc = dataset[:, 14:17]
    angular_pos = dataset[:,17:20]
    states = np.hstack((linear_pos, angular_pos, linear_vel, angular_vel, linear_acc, angular_acc))
    controls = dataset[:, 29:33]

    # states = torch.tensor(np.array(states, dtype=np.float32))
    # controls = torch.tensor(np.array(controls, dtype=np.float32))
    # dt_test = torch.tensor(np.array(dt_test, dtype=np.float32))

    return states, controls, dt_test


def normalize_NN_inputs(X_current=None, U_curr=None, mean=None, std=None):
    """
    Normalize NN inputs using provided mean and std.
    """
    # Unpack mean and std
    lin_pos_mean, lin_vel_mean, ang_vel_mean, lin_acc_mean, ang_acc_mean, controls_mean = mean[:3], mean[3:6], mean[6:9], mean[9:12], mean[12:15], mean[15:19]
    lin_pos_std, lin_vel_std, ang_vel_std, lin_acc_std, ang_acc_std, controls_std = std[:3], std[3:6], std[6:9], std[9:12], std[12:15], std[15:19]

    # Normalize current states
    linear_pos_curr_norm = (X_current[:, :3] - lin_pos_mean) / lin_pos_std
    linear_vel_curr_norm = (X_current[:, 6:9] - lin_vel_mean) / lin_vel_std
    angular_vel_curr_norm = (X_current[:, 9:12] - ang_vel_mean) / ang_vel_std
    # linear_acc_curr_norm = (X_current[:, 12:15] - lin_acc_mean) / lin_acc_std
    # angular_acc_curr_norm = (X_current[:, 15:18] - ang_acc_mean) / ang_acc_std
    controls_curr_norm = (U_curr - controls_mean) / controls_std

    # Reconstruct normalized current state tensor
    X_current_norm = np.hstack((linear_pos_curr_norm, X_current[:, 3:6], linear_vel_curr_norm, angular_vel_curr_norm)) #, linear_acc_curr_norm, angular_acc_curr_norm))
    U_curr_norm = controls_curr_norm

    return X_current_norm, U_curr_norm


def load_XU_data():
    # 2) Load dataset
    dataset_rough = pd.read_csv('dataset/data_set_drone.csv')
    # Print the type of each column
    # print(f"data_set_drone columns types: {dataset_rough.dtypes}")

    # 2a) I want to lead back the new dataset to the previous format (of the old dataset in order to not modify the rest of the code)
    dataset_masked = dataset_masking(dataset_rough)

    # 2b) Save the new dataset
    # dataset_masked.to_csv('dataset/dataset_masked.csv', index=False)
    #print(f"Cmd_thrust: min={dataset['cmd_thrust'].min()}, max={dataset['cmd_thrust'].max()}")

    mass = 2.0
    inertia = np.array([0.0217, 0.0217, 0.04])
    g = 9.81

    # 3) Preprocess of the dataset
    # 3a) Complete the dataset with the missing information
    dataset = np.array(dataset_masked)
    dataset = from_quaternion_to_euler(dataset)
    dataset = px4_pwm_to_thrust(dataset = dataset, mass=mass, g=g)  # Convert PX4 signals (cmd_thrust (probably [PWM])) to cmd_thrust values [N]
    dataset = px4_angular_rate_to_torque(dataset= dataset, inertia = inertia)  # Convert PX4 angular rates (cmd_bodyrates (probably [rad/s])) to torques [N*m]


    # pd.DataFrame(dataset).to_csv('dataset/dataset.csv', index=False)

    # 3b) Normalize the dataset for training and validation (testing dataset will not be normalized)
    m, _ = dataset.shape
    data_mean, data_std = get_mean_and_std(dataset)


    # 4) define the dt as the average of all the dt in the dataset
    dt_all = dataset[:, 1:2]  # Assuming the last column contains the time intervals
    dt_mean = np.mean(dt_all)

    # Configure dataset to represent its values
    X_data, U_data, dt_data = configure_data(dataset)

    return X_data, U_data, data_mean, data_std, dt_mean



def TrainValTest_split(X_data, U_data, data_mean, data_std, dt_mean, SEED):
    # Split X_data and U_data into training, validation, and testing sets (70% train, 25% val, 5% test)

    set_seed(SEED)

    train_percentage = 0.7418693231760913
    val_percentage = 0.2471432757105186
    test_percentage = 0.010987401113389979

    if not np.isclose(train_percentage + val_percentage + test_percentage, 1.0):
        raise ValueError("Train, validation, and test percentages must sum to 1.0")

    X_data_current = X_data[:-1]
    X_data_next = X_data[1:]
    U_data_current = U_data[:-1]
    U_data_next = U_data[1:]        # Not needed

    X_train_val_current, X_test_current, X_train_val_next, X_test_next, U_train_val_current, U_test_current, U_train_val_next, U_test_next = train_test_split(
            X_data_current,
            X_data_next,
            U_data_current,
            U_data_next,
            test_size=test_percentage, 
            random_state=SEED, 
            shuffle=True) # 5% of the total dataset for testing

    val_percentage_adjusted = val_percentage / (train_percentage + val_percentage)

    X_train_current, X_val_current, X_train_next, X_val_next, U_train_current, U_val_current, U_train_next, U_val_next = train_test_split(
            X_train_val_current,
            X_train_val_next,
            U_train_val_current,
            U_train_val_next,
            test_size=val_percentage_adjusted,
            random_state=SEED,
            shuffle=True
        )


    X_train_current_norm, U_train_current_norm = normalize_NN_inputs(X_train_current, U_train_current, data_mean, data_std)
    X_val_current_norm, U_val_current_norm = normalize_NN_inputs(X_val_current, U_val_current, data_mean, data_std)
    X_test_current_norm, U_test_current_norm = normalize_NN_inputs(X_test_current, U_test_current, data_mean, data_std)

    X_train_next_norm, U_train_next_norm = normalize_NN_inputs(X_train_next, U_train_next, data_mean, data_std)
    X_val_next_norm, U_val_next_norm = normalize_NN_inputs(X_val_next, U_val_next, data_mean, data_std)
    X_test_next_norm, U_test_next_norm = normalize_NN_inputs(X_test_next, U_test_next, data_mean, data_std)

    X_train_current = torch.tensor(X_train_current, dtype=torch.float32)
    X_train_next = torch.tensor(X_train_next, dtype=torch.float32)
    X_train_current_norm = torch.tensor(X_train_current_norm, dtype=torch.float32)
    X_train_next_norm = torch.tensor(X_train_next_norm, dtype=torch.float32)
    U_train_current = torch.tensor(U_train_current, dtype=torch.float32)
    U_train_next = torch.tensor(U_train_next, dtype=torch.float32)
    U_train_current_norm = torch.tensor(U_train_current_norm, dtype=torch.float32)
    U_train_next_norm = torch.tensor(U_train_next_norm, dtype=torch.float32)
    X_val_current = torch.tensor(X_val_current, dtype=torch.float32)
    X_val_next = torch.tensor(X_val_next, dtype=torch.float32)
    X_val_current_norm = torch.tensor(X_val_current_norm, dtype=torch.float32)
    X_val_next_norm = torch.tensor(X_val_next_norm, dtype=torch.float32)
    U_val_current = torch.tensor(U_val_current, dtype=torch.float32)
    U_val_next = torch.tensor(U_val_next, dtype=torch.float32)
    U_val_current_norm = torch.tensor(U_val_current_norm, dtype=torch.float32)
    U_val_next_norm = torch.tensor(U_val_next_norm, dtype=torch.float32)
    # X_test = torch.tensor(X_test, dtype=torch.float32)    # Should remain as numpy
    # U_test = torch.tensor(U_test, dtype=torch.float32)    # Should remain as numpy

    data_to_save = {
        "X_train_current": X_train_current,
        "X_train_next": X_train_next,
        "X_train_current_norm": X_train_current_norm,
        "X_train_next_norm": X_train_next_norm,
        "U_train_curr": U_train_current,
        "U_train_next": U_train_next,
        "U_train_curr_norm": U_train_current_norm,
        "U_train_next_norm": U_train_next_norm,
        "X_val_current": X_val_current,
        "X_val_next": X_val_next,
        "X_val_current_norm": X_val_current_norm,
        "X_val_next_norm": X_val_next_norm,
        "U_val_curr": U_val_current,
        "U_val_next": U_val_next,
        "U_val_curr_norm": U_val_current_norm,
        "U_val_next_norm": U_val_next_norm,
        "X_test": X_test_current,  # Should remain as numpy
        "X_test_next": X_test_next,
        "U_test": U_test_current,  # Should remain as numpy
        "U_test_next": U_test_next,
        "dt_mean": dt_mean,
        "data_mean": data_mean,
        "data_std": data_std,
    }

    with open(f"dataset/configured_data_new_{SEED}.pkl", 'wb') as f:
        pickle.dump(data_to_save, f)

def loadSeedData(SEED):
    with open(f"dataset/configured_data_new_{SEED}.pkl", 'rb') as f:
        data_loaded = pickle.load(f)
        X_train_current = data_loaded["X_train_current"]
        X_train_next = data_loaded["X_train_next"]
        X_train_current_norm = data_loaded["X_train_current_norm"]
        X_train_next_norm = data_loaded["X_train_next_norm"]
        U_train_curr = data_loaded["U_train_curr"]
        U_train_next = data_loaded["U_train_next"]
        U_train_curr_norm = data_loaded["U_train_curr_norm"]
        U_train_next_norm = data_loaded["U_train_next_norm"]
        X_val_current = data_loaded["X_val_current"]
        X_val_next = data_loaded["X_val_next"]
        X_val_current_norm = data_loaded["X_val_current_norm"]
        X_val_next_norm = data_loaded["X_val_next_norm"]
        U_val_curr = data_loaded["U_val_curr"]
        U_val_next = data_loaded["U_val_next"]
        U_val_curr_norm = data_loaded["U_val_curr_norm"]
        U_val_next_norm = data_loaded["U_val_next_norm"]
        X_test = data_loaded["X_test"]
        X_test_next = data_loaded["X_test_next"]
        U_test = data_loaded["U_test"]
        U_test_next = data_loaded["U_test_next"]
        dt_mean = data_loaded["dt_mean"]
        data_mean = data_loaded["data_mean"]
        data_std = data_loaded["data_std"]
        # print(f"Data loaded successfully from pickle file configured_data_new_{SEED}.pkl")
    return [X_train_current, X_train_next, X_train_current_norm, X_train_next_norm, U_train_curr, U_train_next, U_train_curr_norm, U_train_next_norm, X_val_current, X_val_next, X_val_current_norm, X_val_next_norm, U_val_curr, U_val_next, U_val_curr_norm, U_val_next_norm, X_test, X_test_next, U_test, U_test_next, dt_mean, data_mean, data_std]