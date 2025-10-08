import numpy as np

np.set_printoptions(precision=2, suppress=True, linewidth=200)

def create_1d_gaussian_array(size, mean=1, sigma=1):
    """
    Return 1D array of Gaussian-shaped values.
    """
    x = np.linspace(0, size-1, size)
    gaussian_values = (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-(x-mean)**2 / 2*sigma**2)
    return gaussian_values


# a 2D matrix, each row is a state, each column is a infinitesimal timestamp
# each state is gaussian distributed and tiled together by the same amount of shift in time 
num_states = 10
num_timestamps = 20
state_size = 5
initial_mean = 0
std_dev = 1
shift = 2

matrix = np.zeros((num_states, num_timestamps))
print(matrix)

current_start_col = 0
for i in range(num_states):
    if current_start_col >= num_timestamps:
        break

    state_values = np.random.normal(loc=initial_mean, scale=std_dev, size=state_size)
    end_col = min(current_start_col + state_size, num_timestamps)
    length = end_col-current_start_col
    matrix[i, current_start_col:end_col] = create_1d_gaussian_array(length)
    
    current_start_col += shift

print(matrix)
