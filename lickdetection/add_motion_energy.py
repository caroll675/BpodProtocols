# plot the motion energy trace on the video 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

motion_energy_file = r"C:\video\CamData_combined\CC4_20251104_145442_cam1.mat"
video_file = r"\\140.247.90.110\homes2\Carol\LickData\annotated_video\CC4_20251010_annotated.mp4"
motion_energy = sio.loadmat(motion_energy_file)['CamData']['mot_energy'][0,0] # Accessing the correct structure
labels = {0:'WhiskerPad', 1:'Nose', 2:'Tongue', 3:'Eye'}
nose_trace = motion_energy[:,1]

# Initialize matplotlib figure for the plot
fig, ax = plt.subplots(figsize=(6, 2), dpi=100)
line, = ax.plot([], [], color='blue')
ax.set_xlim(0, len(nose_trace))
ax.set_ylim(np.min(nose_trace), np.max(nose_trace) * 1.1)
ax.set_facecolor('black') # Set background to transparent for better contrast
fig.patch.set_facecolor('black') # Set figure background
ax.tick_params(axis='x', colors='white') # Set tick color to white
ax.tick_params(axis='y', colors='white') # Set tick color to white
ax.spines['left'].set_color('white')
ax.spines['bottom'].set_color('white')
ax.spines['right'].set_color('white')
ax.spines['top'].set_color('white')
ax.set_title('Nose Motion Energy', color='white')

# plot the nose motion energy trace on the video like animation 
video = cv2.VideoCapture(video_file)
frame_num = 0
window_size = 200 # Define a smaller window size for the plot
while True:
    ret, frame = video.read()
    if not ret:
        break
    
    # Determine the start and end indices for the rolling window
    plot_start_idx = max(0, frame_num - window_size)
    plot_end_idx = frame_num
    
    # Update plot
    line.set_data(np.arange(plot_start_idx, plot_end_idx), nose_trace[plot_start_idx:plot_end_idx])
    ax.set_xlim(plot_start_idx, plot_start_idx + window_size) # Set x-lim to the rolling window
    # ax.autoscale_view() # No longer needed with fixed x-lim
    
    # Render matplotlib figure to an image
    fig.canvas.draw()
    plot_image = np.array(fig.canvas.renderer._renderer) # This line might need adjustment depending on matplotlib backend
    plot_image = cv2.cvtColor(plot_image, cv2.COLOR_RGB2BGR)
    
    # Resize plot image to fit the frame
    plot_height, plot_width, _ = plot_image.shape
    frame_height, frame_width, _ = frame.shape
    
    # Define position for the plot overlay
    # For example, top-left corner with some padding
    x_offset = 10
    y_offset = 300
    
    # Ensure plot_image dimensions don't exceed frame dimensions at the offset
    effective_plot_height = min(plot_height, frame_height - y_offset)
    effective_plot_width = min(plot_width, frame_width - x_offset)

    if effective_plot_height > 0 and effective_plot_width > 0:
        resized_plot_image = cv2.resize(plot_image, (effective_plot_width, effective_plot_height))
        frame[y_offset:y_offset+effective_plot_height, x_offset:x_offset+effective_plot_width] = resized_plot_image

    cv2.imshow('Frame', frame)
    
    # Increment frame number
    frame_num += 1
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()

