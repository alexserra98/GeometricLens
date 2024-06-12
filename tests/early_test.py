# %%
import numpy as np
# %%
import numpy as np

def concatenate_layers(input_array, window_size):
    # Ensure input_array is a numpy array
    if not isinstance(input_array, np.ndarray):
        input_array = np.array(input_array)
    
    # Prepare output array
    num_windows = input_array.shape[1] - window_size + 1
    output_shape = (input_array.shape[0], input_array.shape[1], window_size * input_array.shape[2])
    output = np.zeros(output_shape, dtype=input_array.dtype)
    
    # Create all windows for each position that fits the full window size
    windows = np.lib.stride_tricks.sliding_window_view(input_array, (1, window_size, input_array.shape[2]))
    windows = windows.reshape(input_array.shape[0], num_windows, -1)
    
    # Assign these windows to the output
    output[:, :num_windows] = windows
    
    # Handling the last layers by concatenating backwards
    # We need to handle the case where the indices fall out of the bounds normally handled by the first loop
    if window_size > 1:
        for i in range(num_windows, input_array.shape[1]):
            output[:, i, :] = input_array[:, i - window_size + 1:i + 1].reshape(input_array.shape[0], -1)
    
    return output

test_input = np.random.rand(10, 10, 10)
test_output = concatenate_layers(test_input, window_size=2)

assert np.allclose(test_output[0,2],test_input[0,2:4].reshape(-1))
