import os
import pickle
import numpy as np

# pickle_dir = "dataset_multipendulum_gaussian/saved_pickles"
# pickle_dir = "dataset_pendulum/picklefolder_uniform_sameinitcond"
# pickle_dir = "dataset_pendulum_bigrange/picklefolder"
# pickle_dir = "dataset_pendulum/picklefolder_testing_model"
# pickle_dir = "dataset_pendulum/picklefolder"
# pickle_dir = "/data/esmith/Dataset_LinearSystem_ICL/picklefolder"
# pickle_dir = "/data/esmith/Dataset_LinearSystem_ICL/dagger_picklefolder_0"
pickle_dir = "/data/esmith/Dataset_Cartpole_ICL/picklefolder"
# pickle_dir = "/data/esmith/dataset_pendulum/picklefolder_testing_model"

file_number = 0 #205000 

# pickle_file = f"multipendulum_{file_number}.pkl"
# pickle_file = f"multipendulum_{file_number}.pkl"
pickle_file = f"batch_{file_number}.pkl"

file_path = os.path.join(pickle_dir, pickle_file)

if not os.path.exists(file_path):
    print(f"File '{pickle_file}' does not exist in '{pickle_dir}'.")
else:
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        
        print(f"Contents of {pickle_file}:")
        # print(np.shape(np.squeeze(data[0]).cpu().detach().numpy()))
        # print(np.shape(np.squeeze(data[1])))
        # print(data[0][1])
        # print(data[0][:,5,:])
        print(np.shape(data[0]))
        print(np.shape(data[1]))

        # print(data[0][0])
        # print(data[1][0])
        # print(np.shape(data[2]))
        # print(np.shape(data[3]))
        # print(data[2][0])
        # print(data[3][0])
        # print(data[0])
        # print(data[1])
    except Exception as e:
        print(f"An error occurred while reading {pickle_file}: {e}")
