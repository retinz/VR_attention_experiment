import csv
from psychopy import visual, core, event, gui
import os
import random
import numpy as np
import time
from PIL import Image
import pandas as pd
import python_markers.marker_management as mark
import python_markers.GS_timing as timing
from openpyxl import load_workbook
from datetime import datetime
from collections import OrderedDict

# Runs a simple break
def run_break(win, duration, break_marker_start):
    """
    Runs a break with specified duration and markers for start and end.
    
    Parameters:
        duration (int): Duration of the break in seconds.
        break_marker_start (int): Marker value to signal the start of the break.
    """
    # Display initial message for ECG baseline recording
    screen_width, screen_height = win.size
    dynamic_font_size = 28 * (screen_height / 800)

    # Duration conversion
    minutes = duration // 60

    # Start timer for ECG recording
    start_time = core.getTime()

    # Break start marker
    marker_manager.set_value(break_marker_start)
    timing.delay(100)
    marker_manager.set_value(0)

    message_text = (f"A break for {minutes} min. Please stay seated.")
    message = visual.TextStim(win, text=message_text, color="white", height=dynamic_font_size, wrapWidth=0.7 * screen_width)
    message.draw()
    win.flip()

    # Wait for the specified duration
    while core.getTime() - start_time < duration:
        check_quit()
        core.wait(0.1)

    # Break end marker
    marker_manager.set_value(break_marker_start + 1)
    timing.delay(100)
    marker_manager.set_value(0)

# Modular dialog function using DlgFromDict for showing a dialog and handling "Cancel" press
def show_dialog(dialog_data, title=""):
    # Display dialog using DlgFromDict, which automatically creates fields from dictionary keys
    dialog = gui.DlgFromDict(dictionary=dialog_data, title=title, sortKeys=False)

    # If "Cancel" is pressed, quit the experiment
    if not dialog.OK:
        core.quit()
    
    # Returns the updated dictionary with user inputs
    return dialog_data

# Save consent data as CSV, with headers derived from the dictionary keys
def save_consent_data(file_path, consent_data):
    headers = list(consent_data.keys())  # Derive headers from dictionary keys
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='', encoding='utf-8') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=headers)
        if not file_exists or os.path.getsize(file_path) == 0:
            dict_writer.writeheader()
        dict_writer.writerow(consent_data)

# Start-Up Information ---------------------------------------------
def generate_consent_screen(win):
    # Define the file path and headers for saving consent data
    file_path = paths['informed_consent']

    # Screen dimensions for dynamic sizing
    screen_width, screen_height = win.size
    def_dynamic_font_size = 20 * (screen_height / 800)
    options_font_size = 25 * (screen_height / 800)
    title_height = 0.035 * screen_height
    padding = 50

    # Text to be displayed
    title_text = "INFORMED CONSENT"
    intro_text = (
        "for participation in the scientific research experiment:\n\n"
        "Test your Attention Skills II\n\n"
        "The entire experiment will take approximately two and a half hours, divided into two sessions.\n\n"
        "You have the right to stop the experiment at any time. Your data will be processed anonymously.\n\n"
    )
    bullet_points_text = (
        "• I have been satisfactorily informed about the research.\n"
        "• I have read the written information properly.\n"
        "• I was given the opportunity to ask questions about the research.\n"
        "• My questions have been answered satisfactorily.\n"
        "• I have thought carefully about participating in the study.\n"
        "• I have the right to withdraw my consent at any time, without having to give a reason.\n\n"
    )
    options_text = "I agree to participate in the study.\n\n Press 'y' for YES, 'n' for NO"

    # Display consent content
    title = visual.TextStim(win, text=title_text, pos=(0, 0.35 * screen_height), color="white", height=title_height, wrapWidth=0.6 * screen_width)
    item_box = visual.Rect(win, width=title.boundingBox[0] + padding, height=title.boundingBox[1] + padding,
                           pos=(0, 0.35 * screen_height), lineColor="white", lineWidth=4, fillColor=None)
    intro = visual.TextStim(win, text=intro_text, pos=(0, 0.15 * screen_height), color="white", height=def_dynamic_font_size, wrapWidth=0.7 * screen_width)
    bullet_points = visual.TextStim(win, text=bullet_points_text, pos=(0.20, -0.10 * screen_height), color="white", height=def_dynamic_font_size, wrapWidth=0.7 * screen_width, alignText="left")
    options = visual.TextStim(win, text=options_text, pos=(0, -0.32 * screen_height), color="white", height=options_font_size, wrapWidth=0.7 * screen_width)

    # Draw and display all elements
    title.draw(); item_box.draw(); intro.draw(); bullet_points.draw(); options.draw()
    win.flip()

    # Wait for response
    response = event.waitKeys(keyList=['y', 'n'])

    # If consent is given (y pressed), proceed with dialogs
    if response[0] == 'y':
        consent_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Go to windowed mode
        win.close()  # Close any open windowed instance
        win = visual.Window(fullscr=False, size=(1280, 720), color=[-1, -1, -1], units='pix')
        win.mouseVisible = True

        # Show dialog for participant info with ordered fields
        dialog_data = OrderedDict([
            ("First Name", ""),
            ("Surname", ""),
            ("Date of Birth", "")
        ])
        participant_info = show_dialog(dialog_data, title="Consent Information")

        # Retrieve participant details from dialog
        first_name = participant_info["First Name"]
        surname = participant_info["Surname"]
        dob = participant_info["Date of Birth"]

        # Prepare consent data for saving as a dictionary
        consent_data = {
            "first_name": first_name,
            "surname": surname,
            "dob": dob,
            "consent_timestamp": consent_timestamp
        }
        
        # Save consent data, ensuring headers are present if missing
        save_consent_data(file_path, consent_data)

        return True, first_name, surname, dob, consent_timestamp, win

    # End experiment if consent is not given
    win.close()
    core.quit()
    return False, None, None, None, None

# Get the environment based on participant ID and session number
def get_environment(database_path, participant_id, session_num):
    # Load the database file
    df = pd.read_excel(database_path, sheet_name='randomisation')

    # Find the row corresponding to the participant ID
    participant_row = df[df['participant_id'] == int(participant_id)]

    if participant_row.empty:
        raise ValueError("Participant ID not found in the database.")
    
    # Retrieve the environment based on the session number
    if session_num == '1':
        environment = participant_row['session_1'].values[0]
    elif session_num == '2':
        environment = participant_row['session_2'].values[0]
    else:
        raise ValueError("Invalid session number. Please enter 1 or 2.")

    return environment

# Function to store additional participant info in CSV format, with headers from dictionary keys
def store_participant_info(participant_info, participant_file_path):
    # Convert list of tuples to dictionary
    participant_info_dict = dict(participant_info)
    headers = list(participant_info_dict.keys())
    file_exists = os.path.isfile(participant_file_path)
    
    with open(participant_file_path, 'a', newline='', encoding='utf-8') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=headers)
        if not file_exists or os.path.getsize(participant_file_path) == 0:
            dict_writer.writeheader()
        dict_writer.writerow(participant_info_dict)

# Main function to get participant info, environment, and store data if session 1
def get_participant_info(first_name, surname, dob, consent_timestamp):

    while True:
        # Prepare initial dialog data
        info_initial = {'Participant ID': '', 'Session Number': '', 'Environment': ['nature', 'urban']}
        # Show initial dialog for Participant ID and Session Number
        participant_info = show_dialog(info_initial, title="Participant Information (Basic)")

        participant_id = participant_info['Participant ID']
        session_num = participant_info['Session Number']
        environment_external = participant_info['Environment']
            
        # Set up paths and access the database
        database_path = paths['database']

        # Get the environment from the database
        environment = get_environment(database_path, participant_id, session_num)
        
        # Check if entered environment matches the database environment
        if environment == environment_external:
            break
        else:
            # Show an error dialog for mismatched data
            error_dialog = gui.Dlg(title="Error")
            error_dialog.addText("Mismatch in the entered data.")
            error_dialog.show()

    # If session is 1, open a second dialog for additional information
    if session_num == '1':
        # Prepare additional dialog data as an OrderedDict to enforce field order
        info_additional = OrderedDict([
            ('Sex', ['Male', 'Female', 'Other']),  # Dropdown for Sex
            ('Vision', ['Normal', 'Corrected - Contact Lenses', 'Corrected - Eyeglasses'])  # Dropdown for Vision
        ])
        
        # Show additional information dialog
        additional_info = show_dialog(info_additional, title="Participant Information (Additional)")

        info_full = [
            ('participant_id', participant_id),
            ('consent_timestamp', consent_timestamp),
            ('session_num', session_num),
            ('first_name', first_name),
            ('surname', surname),
            ('sex', additional_info.get('Sex')), 
            ('dob', dob),
            ('vision', additional_info.get('Vision'))
        ]

        # Store the detailed information if session is 1
        participant_file_path = paths['participant_info']
        store_participant_info(info_full, participant_file_path)

    # Return the participant ID, session number, and environment
    return participant_id, session_num, environment

# Markers -----------------------------------------------------
# Find the address and make the marker_manager object:
marker_device_type = 'UsbParMarker'
device_info = mark.find_device(device_type=marker_device_type, fallback_to_fake=True)
marker_address = device_info['com_port']
marker_manager = mark.MarkerManager(marker_device_type, marker_address, crash_on_marker_errors=False)

# Marker stats
def get_marker_stats(participant_id, session_num):
    # Generate marker tables
    marker_table, marker_summary, errors = marker_manager.gen_marker_table()
    
    # Define the filename using the participant ID and session number
    filename = f"participant_{participant_id}_session_{session_num}_{environment}_markers.tsv"
    
    # Define the full file path using the "results" directory in `paths`
    file_path = os.path.join(paths["results"], filename)
    
    # Save the main marker table as a TSV file
    with open(file_path, 'w', newline='') as file_out:
        file_out.write("# Summary Table\n")
        marker_summary.to_csv(file_out, sep='\t', index=False)
        file_out.write("\n# Marker Table\n")
        marker_table.to_csv(file_out, sep='\t', index=False)
        file_out.write("\n# Error Table\n")
        errors.to_csv(file_out, sep='\t', index=False)

# Function for ECG and eyetracking baseline recording (ECG: 5 mins, eyetracker: last 2 mins)
def run_baseline(win, eyetracker, csv_writer_eye, order, block_num, ECG_start_marker, eye_start_marker):

    # Display initial message for ECG baseline recording
    screen_width, screen_height = win.size
    dynamic_font_size = 28 * (screen_height / 800)

    show_text(win, text=
        "We will now record baseline heart rate and pupil size. This will take 7 minutes in total. \n\n"
        "During the first 5 minutes, we will be recording heart rate.\n\n" 
        "For the last 2 minutes, we will be recording pupil size.\n\n"
        "Please try not to move much during the whole recording.\n\n"
        "Press the 'Spacebar' to start the heart rate recording."
        )

    message_text = ("Recording heart rate baseline (5 min). Please stay seated.")
    message = visual.TextStim(win, text=message_text, color="white", height=dynamic_font_size, wrapWidth=0.7 * screen_width)
    message.draw()
    win.flip()

    # Start timer for ECG recording
    start_time_ECG = core.getTime()

    # Start ECG recording marker
    marker_manager.set_value(ECG_start_marker)  # Send START marker for ECG
    timing.delay(100)
    marker_manager.set_value(0)
    
    # Wait for 5 minutes (300 seconds)
    while core.getTime() - start_time_ECG < 300:
        check_quit()
        core.wait(0.1)

    # End ECG recording marker
    marker_manager.set_value(ECG_start_marker + 1)  # Send END marker for ECG
    timing.delay(100)
    marker_manager.set_value(0)

    # Eye data baseline recording instructions
    show_text(win, text=
        "We will now switch to recording pupil size. A fixation point will appear in the center of the next screen.\n\n"
        "Please focus on this fixation point for the entire duration of the recording (2 minutes). It is okay to blink as needed.\n\n"
        "Position your head on the CHINREST, ensuring that your forehead rests against the upper bar. Maintain this position throughout the recording.\n\n"
        "When you are ready, press the 'Spacebar' to proceed to the next screen and begin the recording.")
  
    draw_fixation(win)  # Draw fixation point on the screen
    win.flip()
    
    # Eyetracking marker start
    marker_manager.set_value(eye_start_marker)
    timing.delay(100)
    marker_manager.set_value(0)
    
    # Start eyetracking data recording
    subscribe_eyetracker(eyetracker, csv_writer_eye, order, block_num)
    
    start_time_eye = core.getTime()
    
    # Continue recording for 1 more minutes (60 seconds)
    while core.getTime() - start_time_eye < 120:
        check_quit()
        core.wait(0.1)

    # Stop eyetracking data recording
    unsubscribe_eyetracker(eyetracker)

    # Eyetracker end marker
    marker_manager.set_value(eye_start_marker + 1) 
    timing.delay(100)
    marker_manager.set_value(0)

    # Display message indicating recording is complete
    end_message = visual.TextStim(win, text="Recording complete. You may now remove your chin from the chinrest.\n\n Press the 'Spacebar' to continue.", color="white", height=dynamic_font_size, wrapWidth=0.7 * screen_width)
    end_message.draw()
    win.flip()
    
    # Wait for participant to press space to continue
    event.waitKeys(keyList=["space"])

# Escape ---------------------------------------------------------------
# Check for "Ctrl + Alt + Q" combination to quit the experiment
def check_quit():
    keys = event.getKeys(modifiers=True)
    for key, mod in keys:
        if key == 'q' and mod['ctrl'] and mod['alt']:
            core.quit()  # End the experiment

# Path management ----------------------------------------------------
# Utility function to normalize paths (cross-platform compatibility)
def normalize_path(path):
    return os.path.normpath(path)  # Use os.path.normpath to handle mixed slashes

# Function to set up file paths based on the main folder
def setup_paths(main_folder):
    # Normalize the main folder path
    main_folder = normalize_path(main_folder)
    
    # Define subfolder paths relative to the main folder, and normalize them
    paths = {
        'results': normalize_path(os.path.join(main_folder, "results")),
        'images': normalize_path(os.path.join(main_folder, "images")),
        'VR_rating': normalize_path(os.path.join(main_folder, "VR_rating")),
        'database': normalize_path(os.path.join(main_folder, "VR_rating", "database.xlsx")),  # This points to the Excel file
        'participant_info': normalize_path(os.path.join(main_folder, "participants", "participant_info.csv")),
        'informed_consent': normalize_path(os.path.join(main_folder, "participants", "informed_consent.csv"))
    }

    return paths

# AST block statistics --------------------------------------------------
def show_block_statistics(win, trial_results, block_num):
    # Get screen dimensions for dynamic sizing
    screen_width, screen_height = win.size

    # Calculate dynamic font size (same as in show_text)
    dynamic_font_size = 28 * (screen_height / 800)  # Adjust font size based on screen height

    # Calculate the average response time in milliseconds for correct responses only
    response_times = [trial['response_time'] for trial in trial_results 
                      if trial['response_time'] is not None and trial['response'] == 'correct']
    
    if response_times:
        avg_response_time_ms = np.mean(response_times) * 1000  # Convert seconds to milliseconds
    else:
        avg_response_time_ms = 'NA'  # No valid correct response times

    # Calculate accuracy (percentage of correct responses)
    correct_trials = [trial for trial in trial_results if trial['response'] == 'correct']
    total_trials = len(trial_results)
    if total_trials > 0:
        accuracy = len(correct_trials) / total_trials * 100  # Accuracy as percentage
    else:
        accuracy = 0.0  # No trials to calculate accuracy

    # Show the average response time and accuracy on the screen
    if isinstance(avg_response_time_ms, (int, float)):
        avg_response_time_str = f"{avg_response_time_ms:.2f} ms"
    else:
        avg_response_time_str = "NA"
    
    # Determine the variable text
    if block_num == 4:
        variable_text = "Great job!"
    else:
        variable_text = "Well done! Let's see if you can be even faster and more accurate in the next block!"

    stats_text = (
        f"Your performance for this block:\n\n"
        f"Average Response Time: {avg_response_time_str}\n"
        f"Accuracy: {accuracy:.2f}%\n\n"
        f"{variable_text}\n\n"
        "Take a break and press the 'Spacebar' to continue."
    )

    # Create the visual text object with dynamic font size
    message = visual.TextStim(win, text=stats_text, color="white", height=dynamic_font_size, wrapWidth=0.7 * screen_width)
    message.draw()
    win.flip()

    # Wait for the participant to press the spacebar to continue
    event.waitKeys(keyList=["space"])

# VR rating [recognition task] (part of manipulation/positive check) ---------------------------------------------
def run_image_recognition(participant_id, session_num, win):
    # Get the screen width and height
    screen_width, screen_height = win.size

    # Step 1: Load the database Excel sheet
    excel_path = paths['database']  # Use the corrected path for the Excel file
    df_database = pd.read_excel(excel_path, sheet_name=0)
    df_answers = pd.read_excel(excel_path, sheet_name=1)  # Sheet with correct answers

    # Step 2: Locate participant's ID and session number in the Excel sheet
    participant_id = int(participant_id)  # Convert participant_id to integer before lookup
    participant_row = df_database[df_database['participant_id'] == participant_id]
    
    if participant_row.empty:
        print(f"Participant ID {participant_id} not found in randomisation sheet.")
        return  # Exit the function if participant ID is not found

    # Step 3: Determine the session type ('nature' or 'urban') based on the session number
    session_column = 'session_1' if str(session_num) in ['1', '01'] else 'session_2'
    session_environment = participant_row[session_column].values[0]
    
    # Step 4: Determine which folder to access based on the session environment
    environment_folder = os.path.join(paths['VR_rating'], session_environment)
    
    # Step 5: Retrieve the six images from the respective folder
    images = [f for f in os.listdir(environment_folder) if f.endswith('.png')]
    selected_images = images[:12]  # Select first 12 images
    random.shuffle(selected_images)  # Shuffle the image order
    
    # Step 6: Display images one by one and collect responses
    results = []
    for img in selected_images:
        img_path = os.path.join(environment_folder, img)
        
        # Open the image to get its original size
        pil_img = Image.open(img_path)
        original_width, original_height = pil_img.size
        
        # Calculate aspect ratio
        aspect_ratio = original_width / original_height
        
        # Define the max width and height for display
        max_width, max_height = 0.5 * win.size[0], 0.7 * win.size[1]  # Set relative to screen size
        
        # Adjust width and height to maintain aspect ratio
        if original_width > original_height:
            new_width = min(original_width, max_width)
            new_height = new_width / aspect_ratio
        else:
            new_height = min(original_height, max_height)
            new_width = new_height * aspect_ratio
        
        # Create the ImageStim with dynamic size
        image_stim = visual.ImageStim(win, image=img_path, interpolate=True, size=(new_width, new_height), pos=(0, 0.00 * win.size[1]))
        
        # Display image with new question and response options
        question_text = visual.TextStim(win, text="How much did you like being in this place?", pos=(0, 0.35 * win.size[1]), color="white", height=0.045 * win.size[1])
        
        # Responses arranged horizontally with numbers on top
        options = [
            ("I don’t recognise this place", -0.34 * win.size[0]),
            ("I disliked being in this place", -0.12 * win.size[0]),
            ("I didn’t like or dislike being in this place", 0.12 * win.size[0]),
            ("I liked being in this place", 0.34 * win.size[0])
        ]
        
        # Draw all the stimuli
        question_text.draw()
        image_stim.draw()
        
        # Draw response options and numbers dynamically
        for i, (option_text, pos_x) in enumerate(options):
            # Draw the number above the text
            number_text = visual.TextStim(win, text=str(i + 1), pos=(pos_x, -0.33 * win.size[1]), color="white", height=30)
            response_text = visual.TextStim(win, text=option_text, pos=(pos_x, -0.4 * win.size[1]), color="white", height=25, wrapWidth=0.20 * win.size[0])

            # Draw number and text
            number_text.draw()
            response_text.draw()

        win.flip()

        # Wait for response (1-4)
        response = None
        while response is None:
            keys = event.getKeys(keyList=['1', '2', '3', '4'])
            if keys:
                response = keys[0]

        # Match image to environment to check if correct
        correct_row = df_answers[df_answers['environment'] == img.split('.')[0]]
        if not correct_row.empty:
            correct_answer = correct_row['correct_response'].values[0]
            is_correct = (response == '1' and correct_answer == 'no') or (response in ['2', '3', '4'] and correct_answer == 'yes')
        else:
            is_correct = False  # Default to incorrect if no matching row found
        
        # Store the result
        results.append({
            'participant_id': participant_id,
            'session_num': session_num,
            'timestamp': get_timestamp(),
            'image': img,
            'response': response,
            'correct': 'correct' if is_correct else 'incorrect'
        })
    
    # Step 7: Save results to the 'results' folder in the main folder
    csv_filename = f"participant_{participant_id}_session_{session_num}_{environment}_VR_rating.csv"
    csv_fullpath = os.path.join(paths['results'], csv_filename)
    
    with open(csv_fullpath, 'w', newline='') as output_file:
        fieldnames = ['participant_id', 'session_num', 'timestamp', 'image', 'response', 'correct']
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

# Timestamp -----------------------------------------------
# Function to get the current timestamp in a readable format with milliseconds
def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # Year-Month-Day Hour:Minute:Second:Millisecond

# Eyetracker ---------------------------------------------------
csv_writer_eye = None # Global variable to hold the CSV writer for eye-tracking data
eyetracker_connected = False

# Attempt to import tobii_research and set eyetracker connection status
try:
    import tobii_research as tr
    eyetracker_connected = True
    print("tobii_research imported successfully.")
except ImportError:
    eyetracker_connected = False
    print("tobii_research package not found. Eyetracker functionality disabled.")

# Function to initialize and connect to the Tobii Spark Pro eyetracker
def initialize_eyetracker():
    global eyetracker_connected
    if eyetracker_connected:
        eyetrackers = tr.find_all_eyetrackers()
        if eyetrackers:
            eyetracker = eyetrackers[0]  # Assume the first tracker found is the one we want
            print("Connected to eyetracker:", eyetracker)
            return eyetracker
        else:
            print("No eyetracker found. Quitting the experiment.")
            eyetracker_connected = False
            core.quit()
    return None

# Function to initialize the eye-tracking CSV file for continuous writing
def initialize_eyetracking_csv(participant_id, session_num):
    """
    Initializes the eye-tracking CSV file for continuous writing. 
    It creates the file if it doesn't exist, writes the header if the file is empty, 
    and returns the CSV writer and file object.
    
    Args:
        participant_id (int/str): The participant's ID.
        session_num (int/str): The session number.
        
    Returns:
        csvfile: The file object for the CSV file.
        csv_writer: A DictWriter object for writing data to the CSV file.
    """
    global csv_writer_eye  # Declare as global to use and modify it here

    # Path for the eye-tracking CSV file
    eyetracking_csv_filename = os.path.join(
        paths['results'], f"participant_{participant_id}_session_{session_num}_{environment}_eyetracking.csv"
    )

    # Normalize the path
    eyetracking_csv_filename = os.path.normpath(eyetracking_csv_filename)

    # Define the field names for the CSV file
    fieldnames = [
        'core_time_stamp', 'device_time_stamp',
        'left_gaze_point', 'left_gaze_validity', 'left_pupil_size', 'left_pupil_validity',
        'right_gaze_point', 'right_gaze_validity', 'right_pupil_size', 'right_pupil_validity', 'order',
        'block_number'
    ]

    # Open the CSV file in append mode
    csvfile_eye = open(eyetracking_csv_filename, 'a', newline='')
    csv_writer_eye = csv.DictWriter(csvfile_eye, fieldnames=fieldnames)

    # Write the header if the file is new (empty)
    if csvfile_eye.tell() == 0:  # File pointer at start, file is empty
        csv_writer_eye.writeheader()

    # Return the open CSV file and writer for further writing during the experiment
    return csvfile_eye, csv_writer_eye

# Function to record the entire eyetracking data sample into the CSV
def get_gaze_data(gaze_data):
    # Manually capture the system and core time at the moment data is received
    core_time_stamp = core.getTime() # Synchronised with the start of the experiment
    # Collect gaze and pupil data from the device
    device_time_stamp = gaze_data['device_time_stamp'] # From Tobii
    
    # Left eye data
    left_gaze_point = gaze_data['left_gaze_point_on_display_area']
    left_gaze_validity = gaze_data['left_gaze_point_validity']
    left_pupil_diameter = gaze_data['left_pupil_diameter']
    left_pupil_validity = gaze_data['left_pupil_validity']
    
    # Right eye data
    right_gaze_point = gaze_data['right_gaze_point_on_display_area']
    right_gaze_validity = gaze_data['right_gaze_point_validity']
    right_pupil_diameter = gaze_data['right_pupil_diameter']
    right_pupil_validity = gaze_data['right_pupil_validity']

    # Return the extracted gaze data as a dictionary
    return {
        'core_time_stamp': core_time_stamp,
        'device_time_stamp': device_time_stamp,
        'left_gaze_point': left_gaze_point,
        'left_gaze_validity': left_gaze_validity,
        'left_pupil_size': left_pupil_diameter,
        'left_pupil_validity': left_pupil_validity,
        'right_gaze_point': right_gaze_point,
        'right_gaze_validity': right_gaze_validity,
        'right_pupil_size': right_pupil_diameter,
        'right_pupil_validity': right_pupil_validity
    }

# Function to handle writing gaze data to CSV
def write_gaze_data_to_csv(csv_writer_eye, gaze_data, order, block_num):
    # Adding block number to the dictionary
    gaze_data['order'] = order
    gaze_data['block_number'] = block_num
    csv_writer_eye.writerow(gaze_data)

# Subscribe to gaze and pupil data
def subscribe_eyetracker(eyetracker, csv_writer_eye, order, block_num):
    # Callback that gets gaze data and writes it to the CSV
    def gaze_data_callback(gaze_data):
        # Get gaze data
        extracted_data = get_gaze_data(gaze_data)
        # Write the data to CSV with only the block number
        write_gaze_data_to_csv(csv_writer_eye, extracted_data, order=order, block_num=block_num)

    # Subscribe to gaze data and use the callback function
    eyetracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, gaze_data_callback, as_dictionary=True)

# Unsubscribe from data
def unsubscribe_eyetracker(eyetracker):
    eyetracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA)

# PANAS questionnaire function ---------------------------------------------------
def run_PANAS(participant_id, session_num, order, win):
    
    csv_PANAS = f"participant_{participant_id}_session_{session_num}_{environment}_PANAS.csv"
    csv_fullpath_PANAS = os.path.join(paths['results'], csv_PANAS)
    
    # Define the PANAS questionnaire items and response options
    questionnaire = [
        {"itemText": "Interested"},
        {"itemText": "Distressed"},
        {"itemText": "Excited"},
        {"itemText": "Upset"},
        {"itemText": "Strong"},
        {"itemText": "Guilty"},
        {"itemText": "Scared"},
        {"itemText": "Hostile"},
        {"itemText": "Enthusiastic"},
        {"itemText": "Proud"},
        {"itemText": "Irritable"},
        {"itemText": "Alert"},
        {"itemText": "Ashamed"},
        {"itemText": "Inspired"},
        {"itemText": "Nervous"},
        {"itemText": "Determined"},
        {"itemText": "Attentive"},
        {"itemText": "Jittery"},
        {"itemText": "Active"},
        {"itemText": "Afraid"}
    ]
    
    response_options = ["Very slightly or not at all", "A little", "Moderately", "Quite a bit", "Extremely"]

    # Dynamically size text elements based on window size
    screen_width, screen_height = win.size
    instruction_height = 0.035 * screen_height # For the instructions before PANAS
    question_height = 0.04 * screen_height # For the instructional text at the top: "Indicate on the keyboard to what extent you feel this way right now."
    item_height = 0.06 * screen_height # For the questionnaire items
    response_height = 0.025 * screen_height
    number_height = 0.035 * screen_height
    padding = 50

    # Add instructional text before starting PANAS
    def show_instruction_text():
        instruction_text = visual.TextStim(win, text=(
            "You will now be presented with a number of words that describe different feelings and emotions.\n\n"
            "For each word, indicate to what extent you feel this way right now, that is, at the present moment, by pressing the respective number on the keyboard.\n\n"
            "There are no right or wrong answers. \n \n "
            "Press the 'Spacebar' to start."), 
            pos=(0, 0), color="white", height=instruction_height, wrapWidth=0.7 * screen_width)
        instruction_text.draw()
        win.flip()

        # Wait for spacebar to continue
        while True:
            keys = event.getKeys()
            if 'space' in keys:
                break

    # Show the instruction text
    show_instruction_text()

    # Open CSV file to append responses with the timestamp
    with open(csv_fullpath_PANAS, 'a', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=['participant_id', 'session_num', 'order', 'item', 'response', 'timestamp'])
        if output_file.tell() == 0:
            dict_writer.writeheader()  # Write header if the file is new

        top_instruction = visual.TextStim(win, text="Indicate on the keyboard to what extent you feel this way right now.", 
                                          pos=(0, 0.32 * screen_height), color="white", height=question_height, wrapWidth=0.6 * screen_width)
        
        for item in questionnaire:
            response = None  

            # Draw the question and options
            top_instruction.draw()

            # Dynamically size and draw item text
            item_text = visual.TextStim(win, text=item['itemText'], pos=(0, 0.08 * screen_height), color="white", height=item_height)

            # Calculate the text width and height using the boundingBox of the text
            text_width, text_height = item_text.boundingBox[0], item_text.boundingBox[1]

            # Create a rectangle with padding around the text
            rect_width = text_width + padding
            rect_height = text_height + padding
            item_box = visual.Rect(win, width=rect_width, height=rect_height, pos=(0, 0.08 * screen_height), 
                                   lineColor="white", lineWidth=4, fillColor=None)

            # Draw the rectangle behind the item text
            item_box.draw()

            # Draw the item text
            item_text.draw()

            # Draw response options dynamically with adjusted spacing and positioning
            option_spacing = 0.15 * screen_width  # Adjusted for screen size
            for j, option in enumerate(response_options):
                pos_x = (j - 2) * option_spacing  # Positioning options symmetrically around the center
                option_text = visual.TextStim(win, text=option, pos=(pos_x, -0.2 * screen_height), color="white", height=response_height)
                option_number = visual.TextStim(win, text=str(j + 1), pos=(pos_x, -0.25 * screen_height), color="white", height=number_height)
                option_text.draw()
                option_number.draw()

            # Flip the window after drawing the question and options
            win.flip()

            # Wait for a key press (1-5)
            while response is None:
                keys = event.getKeys(keyList=['1', '2', '3', '4', '5'])
                if keys:
                    response = keys[0]
            
            timestamp = get_timestamp()

            # Save the response to the CSV file
            dict_writer.writerow({
                'participant_id': participant_id,
                'session_num': session_num,
                'order': order,
                'item': item['itemText'],
                'response': response,
                'timestamp': timestamp
            })

    # Show final message indicating completion
    completion_text = visual.TextStim(win, text="Thank you for your answers!", 
                                      pos=(0, 0), color="white", height=instruction_height)
    completion_text.draw()
    win.flip()

    core.wait(3)  # Wait for 3 seconds before closing

# Function to display regular instructions ------------------------------------
def show_text(win, text, font_size=28):  # Default font size set here
    # Get screen dimensions for dynamic sizing
    screen_width, screen_height = win.size
    dynamic_font_size = font_size * (screen_height / 800)  # Adjust font size based on screen height

    message = visual.TextStim(win, text=text, color="white", height=dynamic_font_size, wrapWidth=0.7 * screen_width)
    message.draw()
    win.flip()
    
    event.waitKeys(keyList=["space"])  # Wait for space bar to continue
    win.flip()

# AST -------------------------------------------
# Function to collect response from participant
def collect_response():
    response = event.waitKeys(keyList=["up", "left"], maxWait=2.0)  # Wait up to 2 seconds for response
    if response is None:
        return None  # No response
    elif response[0] == "up":
        return "vertical"
    elif response[0] == "left":
        return "horizontal"

# Function to run a block of trials and display statistics
def run_singleton_block(win, trials, participant_id, session_num, order, 
    block_name, block_index, block_num=1, is_practice=False, timestamp=None, eyetracker=None, ECG_order_start_marker=None):
    global csv_writer_eye  # Access the global csv_writer_eye
    if is_practice:
        show_text(win, f"{block_name}: {block_index}\n\n There are 4 practice blocks in total. \n\n -- Press the 'Spacebar' to start --")
    else:
        show_text(win, f"{block_name}: Block {block_index}\n\n There are 4 blocks in total. \n\n -- Press the 'Spacebar' to start --")

    trial_results = []  # Store results for this block
    
    # Subscribe to eye-tracker at the start of the block (only once per block)
    if not is_practice and eyetracker_connected:
        subscribe_eyetracker(eyetracker, csv_writer_eye, order, block_num)
    
    check_quit()

    marker_manager.set_value(ECG_order_start_marker) # Send START marker
    timing.delay(100)
    marker_manager.set_value(0)

    min_jitter = 0.750
    max_jitter = 0.950

    for trial_num, trial in enumerate(trials):
        distractor_present = trial['distractor_presence'] == 'present'

        jitter = random.uniform(min_jitter, max_jitter)

        draw_fixation(win)
        win.flip()
        cue_onset = core.getTime()
        core.wait(jitter)

        correct_orientation, common_shape, shape_color, target_shape, distractor_color = draw_singleton_trial(win, trial)
        win.flip()

        array_onset = core.getTime()
        response = collect_response()
        response_log = core.getTime()
        response_time = response_log - array_onset

        correct = response == correct_orientation

        # Append the trial result, including start and end times
        trial_results.append({
            'participant_id': participant_id,
            'session_num': session_num,
            'block_num': block_num,
            'is_practice': is_practice,
            'order': order,
            'trial_num': trial_num + 1,
            'response': 'correct' if correct else 'incorrect',
            'distractor_present': distractor_present,
            'common_shape': common_shape,
            'common_color': shape_color,
            'target_shape': target_shape,
            'distractor_color': distractor_color,
            'array_onset': array_onset, # Stimulus onset
            'response_log': response_log, # End of trial
            'response_time': response_time,
            'cue_onset': cue_onset, # Cue onset
            'jitter': jitter,
            'target_pos': trial['target_pos'],
            'distractor_pos': trial['distractor_pos']
        })

        if not correct:
            feedback_incorrect(win)

    # Unsubscribe the eye-tracker at the end of the block
    if not is_practice and eyetracker_connected:
        unsubscribe_eyetracker(eyetracker)

    marker_manager.set_value(ECG_order_start_marker + 1) # Send END marker
    timing.delay(100)
    marker_manager.set_value(0)

    check_quit()

    show_block_statistics(win, trial_results, block_num)
    return trial_results # Results of the entire block in terms of trials

# Function to generate trials
def generate_trials(AST_blocks_number, target_counts, distractor_counts):
    """
    Generates trials with the following properties:
    - Distractor-present trials are generated first for all blocks.
    - Positions for targets in distractor-present trials are determined using counters and adjacency rules.
    - Equal number of distractor-absent trials are generated.
    - All trials are split into blocks, maintaining 50% distractor-present and 50% distractor-absent trials.
    - Balanced color and shape distribution is maintained.
    """
    
    # Check that AST_blocks_number is a multiple of 4
    if AST_blocks_number % 4 != 0:
        raise ValueError(f"AST_blocks_number ({AST_blocks_number}) must be a multiple of 4.")
    
    trials = []
    num_positions = 8
    max_target_per_pos = (AST_blocks_number * 24) / num_positions
    max_distractor_per_pos = (AST_blocks_number * 12) / num_positions
    
    target_present_trials_num = AST_blocks_number * 24
    distractor_present_trials_num = AST_blocks_number * 12

    # Define color-shape combinations for balanced distribution
    color_shape_combinations = [('green', 'circle'), ('green', 'diamond'), ('red', 'circle'), ('red', 'diamond')]
    trials_per_combination = target_present_trials_num // len(color_shape_combinations)

    # Generate distractor-present trials for all blocks
    distractor_present_trials = []
    color_shape_list_present = color_shape_combinations * (distractor_present_trials_num // len(color_shape_combinations))
    random.shuffle(color_shape_list_present)

    for trial_idx in range(distractor_present_trials_num):
        color, shape = color_shape_list_present[trial_idx]
        
        # Select an available distractor position
        available_distractor_positions = [i for i in range(num_positions) if distractor_counts[i] < max_distractor_per_pos]
        if not available_distractor_positions:
            raise ValueError("No available positions for distractor.")

        distractor_pos = random.choice(available_distractor_positions)
        distractor_counts[distractor_pos] += 1

        distractor_present_trials.append({
            'distractor_presence': 'present',
            'set_size': 8,
            'common_color': color,
            'common_shape': shape,
            'distractor_pos': distractor_pos
        })

    # Assign target positions to distractor-present trials
    for trial in distractor_present_trials:
        distractor_pos = trial['distractor_pos']
        available_target_positions = [
            i for i in range(num_positions)
            if target_counts[i] < max_target_per_pos and i != distractor_pos and abs(i - distractor_pos) != 1 and not (i == 0 and distractor_pos == num_positions - 1) and not (i == num_positions - 1 and distractor_pos == 0)
        ]

        if not available_target_positions:
            raise ValueError("No available positions for target in distractor-present trials.")

        target_pos = random.choice(available_target_positions)
        target_counts[target_pos] += 1
        trial['target_pos'] = target_pos

    # Generate distractor-absent trials
    distractor_absent_trials = []
    color_shape_list_absent = color_shape_combinations * (distractor_present_trials_num // len(color_shape_combinations))
    random.shuffle(color_shape_list_absent)

    for trial_idx in range(distractor_present_trials_num):
        color, shape = color_shape_list_absent[trial_idx]
        
        # Select an available target position
        available_target_positions = [i for i in range(num_positions) if target_counts[i] < max_target_per_pos]
        if not available_target_positions:
            raise ValueError("No available positions for target in distractor-absent trials.")

        target_pos = random.choice(available_target_positions)
        target_counts[target_pos] += 1

        distractor_absent_trials.append({
            'distractor_presence': 'absent',
            'set_size': 8,
            'common_color': color,
            'common_shape': shape,
            'target_pos': target_pos,
            'distractor_pos': None
        })

    # Combine and split trials into blocks
    all_trials = distractor_present_trials + distractor_absent_trials
    random.shuffle(all_trials)

    blocks = []
    trials_per_block = target_present_trials_num // AST_blocks_number
    for block_idx in range(AST_blocks_number):
        block_trials = all_trials[block_idx * trials_per_block:(block_idx + 1) * trials_per_block]
        
        # Ensure 50% distractor-present and 50% distractor-absent in each block
        present_trials = [trial for trial in block_trials if trial['distractor_presence'] == 'present']
        absent_trials = [trial for trial in block_trials if trial['distractor_presence'] == 'absent']
        
        if len(present_trials) != len(absent_trials):
            raise ValueError(f"Block {block_idx + 1} does not have an equal split of present and absent trials.")

        # Ensure balanced colors in each block
        block_colors = [trial['common_color'] for trial in block_trials]
        if block_colors.count('green') != block_colors.count('red'):
            raise ValueError(f"Block {block_idx + 1} does not have balanced colors.")

        blocks.append(block_trials)

    return blocks  # Returns a list of blocks, each containing a balanced set of trials

# Function to save trial results to a CSV file
def save_AST_results(trial_results, participant_id, session_num):

    # Define the file paths using the paths dictionary
    csv_AST = f"participant_{participant_id}_session_{session_num}_{environment}_AST.csv"
    csv_fullpath_AST = os.path.join(paths['results'], csv_AST)

    if not trial_results:
        return  # If there's no trial result, exit the function early

    keys = trial_results[0].keys()
    with open(csv_fullpath_AST, 'a', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        if output_file.tell() == 0:  # Only write header if the file is empty
            dict_writer.writeheader()
        dict_writer.writerows(trial_results)

# Draw fixation point
def draw_fixation(win):
    fixation_outer = visual.Circle(win, radius=10, fillColor="white", lineColor="white", edges=128)
    fixation_inner = visual.Circle(win, radius=2, fillColor="black", lineColor="black", edges=128)
    fixation_outer.draw()
    fixation_inner.draw()

# Feedback for incorrect answers (with internal win.flip())
def feedback_incorrect(win):
    circle = visual.Circle(win, radius=12, fillColor="#aa0000", lineColor="#aa0000")
    circle.draw()
    win.flip()  # Flip inside the function to show the feedback
    core.wait(1.0)  # Keep the delay to show the feedback

# Draw singleton trial (without internal win.flip())
def draw_singleton_trial(win, trial):
    num_shapes = 8
    
    # Get the current window size
    window_width, window_height = win.size

    # Reference size (1024x768) from OpenSesame
    reference_width = 1024
    reference_height = 768

    # Scaling factor for the radius and shapes
    scaling_factor_width = window_width / reference_width
    scaling_factor_height = window_height / reference_height

    # Apply a reduced scaling factor
    scaling_factor = 0.9 * min(scaling_factor_width, scaling_factor_height)

    # Apply the reduced scaling factor to the radius
    scaled_radius = 224 * scaling_factor 

    # Use the scaled radius in the drawing
    radius = scaled_radius

    # Apply the reduced scaling factor to the shape sizes and lines
    original_circle_radius = 46  # Original circle radius
    original_diamond_size = 92   # Original diamond size (width and height)
    original_line_length = 36    # Original line length
    original_line_width = 4      # Original line width

    # Scale the shapes and line element using the reduced scaling factor
    scaled_circle_radius = original_circle_radius * scaling_factor * 1.15
    scaled_diamond_size = original_diamond_size * scaling_factor * 1.1
    scaled_line_length = original_line_length * scaling_factor # Line element inside the shapes
    scaled_line_width = original_line_width * scaling_factor # Line element inside the shapes

    adj_diamond_line_width = scaled_diamond_size * 0.12  # Adjust multiplier for thickness
    adj_circle_line_width = scaled_diamond_size * 0.12  # Adjust multiplier for thickness

    angles = np.linspace(0, 2 * np.pi, num_shapes, endpoint=False)  # positions

    # Colors in RGB format for PsychoPy
    RED = [1, -1, -1]
    GREEN = [-1, 0.38, -0.37]

    # Map the common color string to the actual RGB values
    color_map = {"red": RED, "green": GREEN}
    shape_color = color_map[trial['common_color']]  # Use the trial's specified color for common shapes
    common_shape = trial['common_shape']  # Use the trial's specified common shape

    # Directly assign the target shape as the opposite of the common shape
    target_shape = 'diamond' if common_shape == 'circle' else 'circle'

    # The distractor color is the opposite of the common shape color
    distractor_color = color_map['red'] if trial['common_color'] == 'green' else color_map['green']

    # Use the pre-assigned target and distractor positions
    target_pos = trial['target_pos']
    distractor_pos = trial['distractor_pos']

    correct_orientation = random.choice(['horizontal', 'vertical'])
    distractor_orientation = random.choice(['horizontal', 'vertical'])

    # Draw each shape on the circle
    for i, angle in enumerate(angles):
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)

        # Determine the color and shape for each position
        if i == target_pos:
            shape_color_i = shape_color  # Normal color
            shape_type = target_shape  # Target shape (odd shape)
            current_orientation = correct_orientation
        elif i == distractor_pos:
            shape_color_i = distractor_color
            shape_type = common_shape  # Common shape
            current_orientation = distractor_orientation
        else:
            shape_color_i = shape_color
            shape_type = common_shape
            current_orientation = random.choice(['horizontal', 'vertical'])

        # Draw the shape with the scaled sizes
        if shape_type == 'diamond':

            # Draw the normal diamond shape
            normal_shape = visual.Rect(win, width=scaled_diamond_size, height=scaled_diamond_size, 
                                       lineColor=shape_color_i, lineWidth=adj_diamond_line_width, pos=(x, y), 
                                       ori=45, fillColor=None)
            normal_shape.draw()

            # Define the dimensions and line width for the larger black shape (to mask outer edges)
            black_shape_size = scaled_diamond_size + (adj_diamond_line_width / 2)
            black_line_width = adj_diamond_line_width

            # Draw the black shape slightly larger to hide the outer part of the line
            black_shape = visual.Rect(win, width=black_shape_size, height=black_shape_size, 
                                      lineColor="black", lineWidth=black_line_width, pos=(x, y), 
                                      ori=45, fillColor=None)
            black_shape.draw()

        else:

            # Draw the normal circle shape
            normal_circle = visual.Circle(win, radius=scaled_circle_radius, lineColor=shape_color_i,
                                          lineWidth=adj_circle_line_width, pos=(x, y), fillColor=None, edges=256)
            normal_circle.draw()

            # Define the radius and line width for the larger black circle
            black_circle_radius = scaled_circle_radius + (adj_circle_line_width / 3.7)
            black_line_width = adj_circle_line_width

            # Draw the black circle slightly larger to hide the outer part of the line
            black_circle = visual.Circle(win, radius=black_circle_radius, lineColor="black",
                                         lineWidth=black_line_width, pos=(x, y), fillColor=None, edges=256)
            black_circle.draw()


        # Add horizontal or vertical line inside the shape with the scaled sizes
        if current_orientation == 'vertical':
            line = visual.Rect(win, width=scaled_line_width, height=scaled_line_length, lineColor="white", fillColor="white", pos=(x, y))
        else:
            line = visual.Rect(win, width=scaled_line_length, height=scaled_line_width, lineColor="white", fillColor="white", pos=(x, y))
        line.draw()

    draw_fixation(win)  # Draw the fixation point in the center

    # Return the correct orientation and trial details (no flip here)
    return correct_orientation, trial['common_shape'], trial['common_color'], target_shape, distractor_color

# AST Instructions ------------------------------------------
# Function to display instruction images in sequence with dynamic scaling and smooth rendering
def show_instruction_images(win):

    image_folder = paths['images']  # Image folder for AST instructions

    # Get the screen width and height
    screen_width, screen_height = win.size

    # List of images with their corresponding key expectations
    instruction_sequence = [
        {"image": "1.png", "key": "space"},
        {"image": "2.png", "key": "space"},
        {"image": "3.png", "key": "space"},
        {"image": "4.png", "key": "space"},
        {"image": "5.png", "key": "up"},
        {"image": "6.png", "key": "space"},
        {"image": "7.png", "key": "space"},
        {"image": "8.png", "key": ["a", "s"]}
    ]

    for instruction in instruction_sequence:
        # Construct full path to the image
        image_path = os.path.join(image_folder, instruction["image"])

        # Open the image to get its original size (using PIL)
        pil_img = Image.open(image_path)
        original_width, original_height = pil_img.size

        # Calculate the aspect ratio of the image
        aspect_ratio = original_width / original_height

        # Define the max width and height relative to the screen size (dynamic scaling)
        max_width = 0.9 * screen_width 
        max_height = 0.9 * screen_height 

        # Adjust the width and height while preserving the aspect ratio
        if original_width > original_height:
            new_width = min(original_width, max_width)
            new_height = new_width / aspect_ratio
        else:
            new_height = min(original_height, max_height)
            new_width = new_height * aspect_ratio

        # Create an image stimulus with dynamic size
        instruction_image = visual.ImageStim(win, image=image_path, interpolate=True, size=(new_width, new_height))
        
        # Draw the image to the window
        instruction_image.draw()
        win.flip()

        # Wait for the expected key press(es)
        if isinstance(instruction["key"], list):
            keys = event.waitKeys(keyList=instruction["key"])
            if 'a' in keys:
                # Restart the instructions if 'a' is pressed
                return show_instruction_images(win)
            elif 's' in keys:
                return  # Continue to the practice block
        else:
            event.waitKeys(keyList=[instruction["key"]])  # Wait for the specified key press

# Experiment Flow
def run_experiment(main_folder):
    global csv_writer_eye, paths, environment

    paths = setup_paths(main_folder)

    # Setup the experiment window
    win = visual.Window(fullscr=True, color=[-1, -1, -1], units='pix')
    win.mouseVisible = False

    consent_given, first_name, surname, dob, consent_timestamp, win = generate_consent_screen(win)

    if consent_given:
    # Pass the participant details including first name, surname, dob, and consent timestamp
        participant_id, session_num, environment = get_participant_info(first_name, surname, dob, consent_timestamp)

    # --------------------------------------------------#
                # ECG and EYETRACKER SET-UP
    # --------------------------------------------------#

    show_text(win, "We will now set up our devices (ECG, VR, eye-tracker).")
    check_quit()

    # Restore to fullscreen for the main experiment flow
    win.close()  # Close any open windowed instance
    win = visual.Window(fullscr=True, color=[-1, -1, -1], units='pix')
    win.mouseVisible = False
    win.flip()

    # --------------------------------------------------#
            # BASELINE 1: ECG and EYETRACKER 
    # --------------------------------------------------#

    # Initialize the eyetracker if available
    eyetracker = initialize_eyetracker()
    # CSV for eye-tracker data
    initialize_eyetracking_csv(participant_id, session_num)

    # ECG baseline recording
    run_baseline(win, eyetracker, csv_writer_eye, order=-2, 
        block_num=-2, ECG_start_marker=5, eye_start_marker=30)
    check_quit()

    # --------------------------------------------------#
                        # PANAS 1
    # --------------------------------------------------#

    marker_manager.set_value(50) # PANAS START marker
    timing.delay(100)
    marker_manager.set_value(0)

    # PANAS questionnaire 1 (before AST)
    run_PANAS(participant_id, session_num, order=1, win=win)
    check_quit()

    marker_manager.set_value(51) # PANAS END marker
    timing.delay(100)
    marker_manager.set_value(0)

    # --------------------------------------------------#
                        # AST PRACTICE
    # --------------------------------------------------#

    # AST instructions start
    show_text(win, 
        "You will now perform a Visual Search Task.\n\n" 
        "Please position your head on the CHINREST, ensuring that your forehead rests against the upper bar. Maintain this position throughout the whole Task.\n\n " 
        "Press the 'Spacebar' to see the instructions.")
    check_quit()

    # Use this function in your main flow to show instructions with trial layouts
    show_instruction_images(win)
    check_quit()

    # Initialize counters for target and distractor counts
    target_counts = [0] * 8
    distractor_counts = [0] * 8

    # Generate practice blocks, each with 24 trials
    practice_block_1 = generate_trials(1, target_counts, distractor_counts)  # Practice block 1
    practice_block_2 = generate_trials(2, target_counts, distractor_counts)  # Practice block 2
    practice_block_3 = generate_trials(3, target_counts, distractor_counts)  # Practice block 3
    practice_block_4 = generate_trials(4, target_counts, distractor_counts)  # Practice block 4

    # Generate marker list
    marker_list = [90 + (5 * i) for i in range(12)]

    # Run practice blocks (24 trials each)
    for practice_block_num, practice_trials in enumerate([practice_block_1, practice_block_2, practice_block_3, practice_block_4], 1):
        
        ECG_order_start_marker = marker_list[practice_block_num - 1]

        practice_results = run_singleton_block(
            win, 
            practice_trials, 
            participant_id, 
            session_num,
            order=0,
            block_name="Practice Block",
            block_index=practice_block_num,
            block_num=practice_block_num,
            is_practice=True,
            ECG_order_start_marker=ECG_order_start_marker, # Explicitly mark this as a practice block
        )
        save_AST_results(practice_results, participant_id, session_num)
        check_quit()

    # --------------------------------------------------#
                        # BREAK
    # --------------------------------------------------#

    show_text(win, 
        "It is time to take a short BREAK (3 min) before you proceed with the actual Visual Search Task.\n\n "
        "You may remove your chin from the chinrest for the break.\n\n "
        "Please try not to move much during the break. \n\n"
        "Press the 'Spacebar' to start the break.")
    check_quit()

    run_break(win, duration=180, break_marker_start=160)
    
    # --------------------------------------------------#
                        # AST 1
    # --------------------------------------------------#

    # Initialize counters for target and distractor counts
    target_counts = [0] * 8
    distractor_counts = [0] * 8

    # AST Order 1
    show_text(win, 
        "You will now perform the actual Visual Search Task.\n\n "
        "Please position your head back on the CHINREST, ensuring that your forehead rests against the upper bar. Maintain this position throughout the whole Task.\n\n "
        "Press the 'Spacebar' to continue.")
    check_quit()

    # Pre-VR block (4 blocks of 24 trials each)
    experiment_trials = [generate_trials(block_num, target_counts, distractor_counts) for block_num in range(5,9)] # A list of lists of dictionaries (so a list of lists of trials for a block)
        
    for block_num in range(5, 9):
        pre_trials = experiment_trials[block_num - 5]  # Extract the block trials

        ECG_order_start_marker = marker_list[block_num - 1]

        pre_results = run_singleton_block(
            win, 
            pre_trials, 
            participant_id, 
            session_num,
            order=1,
            block_name="Visual Search Task",
            block_index=block_num - 4,
            block_num=block_num, 
            is_practice=False,  # This is not a practice block 
            eyetracker=eyetracker,
            ECG_order_start_marker= ECG_order_start_marker,
        )
        save_AST_results(pre_results, participant_id, session_num)

        check_quit()

    # Show break before VR viewing
    show_text(win, "You will now get a virtual reality (VR) headset. Please stay engaged with the VR experience for the entire session. Afterwards, we will ask you about your experience.\n\n Please call the experimenter.")
    check_quit()

    # --------------------------------------------------#
                        # PANAS 2
    # --------------------------------------------------#

    marker_manager.set_value(55) # PANAS START marker
    timing.delay(100)
    marker_manager.set_value(0)

    # PANAS questionnaire 2
    run_PANAS(participant_id, session_num, order=2, win=win)
    check_quit()

    marker_manager.set_value(56) # PANAS END marker
    timing.delay(100)
    marker_manager.set_value(0)

    # --------------------------------------------------#
                # EYETRACKER CALIBRATION 2
    # --------------------------------------------------#

    # Second eye-tracker calibration
    show_text(win, "We will now need to calibrate the eye-tracker again. \n\n Please call the experimenter.")
    check_quit()

    # Go to windowed mode
    win.close()  # Close any open windowed instance
    win = visual.Window(fullscr=False, color=[-1, -1, -1], units='pix')
    win.mouseVisible = False
    win.flip()

    # Status
    show_text(win, "After calibration, press the 'Spacebar' to return to the experiment.")
    check_quit()

    # Restore to fullscreen for the main experiment flow
    win.close()  # Close any open windowed instance
    win = visual.Window(fullscr=True, color=[-1, -1, -1], units='pix')
    win.mouseVisible = False
    win.flip()

    # --------------------------------------------------#
                        # AST 2
    # --------------------------------------------------#
    
    # Reshuffle the trials within each block and also the order of the blocks
    reshuffled_trials = []

    # First, shuffle the order of the blocks themselves
    shuffled_blocks = experiment_trials[:]  # Create a copy of the blocks for reshuffling; experiment_trials is a list of lists of dictionaries (so a list of lists of trials for a block)
    random.shuffle(shuffled_blocks)         # Shuffle the order of the blocks; it's a list of lists of trials for a block, still

    # Now, shuffle the trials within each block
    for block_trials in shuffled_blocks: # Iterates over the list of lists of trials for a block
        reshuffled_block = block_trials[:]  # Create a copy of the trials for reshuffling (a copy of the particular list of dictionaries)
        random.shuffle(reshuffled_block)    # Shuffle the trials within the block (shuffles the dictionaires within the list)
        reshuffled_trials.append(reshuffled_block)

    # AST start
    show_text(win, 
        "You will now perform the Visual Search Task again.\n\n"
        "Please position your head on the CHINREST, ensuring that your forehead rests against the upper bar. Maintain this position throughout the whole Task.\n\n "
        "Press the 'Spacebar' to continue.")
    check_quit()

    # Post-VR block (4 blocks of 24 trials)
    for block_num in range(9, 13):
        post_trials = reshuffled_trials[block_num - 9]  # Extract the reshuffled block trials; reshuffled_trials is a list of lists of dictionaries (trials)
            
        ECG_order_start_marker = marker_list[block_num - 1]

        post_results = run_singleton_block(
            win, 
            post_trials, 
            participant_id, 
            session_num,
            order=2,
            block_name="Visual Search Task",
            block_index=block_num - 8,
            block_num=block_num, 
            eyetracker=eyetracker,
            ECG_order_start_marker= ECG_order_start_marker,
        )
        save_AST_results(post_results, participant_id, session_num)
        check_quit() 


    # --------------------------------------------------#
                        # VR RATING
    # --------------------------------------------------#


    # VR rating
    show_text(win, "You will now be asked some questions about your experience in the VR environment. \n\n Indicate your response to each question using the respective key. \n\n Press 'Spacebar' to start.")

    marker_manager.set_value(70) # VR CHECK START marker
    timing.delay(100)
    marker_manager.set_value(0)

    run_image_recognition(participant_id, session_num, win) 

    marker_manager.set_value(71) # VR CHECK END marker
    timing.delay(100)
    marker_manager.set_value(0)


    # --------------------------------------------------#
             # BASELINE 2: ECG and EYETRACKER 
    # --------------------------------------------------#

    # Transition to post-experimental baseline recording
    baseline_text_2 = ("You are ALMOST at the end of the experiment. \n\n"
                        "We will now do just a bit more of baseline recording.\n\n"
                        "You may remove your chin from the chinrest now.\n\n "
                        "Press the 'Spacebar' to see the instructions again.")
    show_text(win, text=baseline_text_2)

    # ECG baseline recording
    run_baseline(win, eyetracker, csv_writer_eye, order=-1, block_num=-1, 
        ECG_start_marker=10, eye_start_marker=35)
    check_quit()

    # Marker printing and closing marker_manager
    get_marker_stats(participant_id, session_num)
    marker_manager.close()

    show_text(win, text="This is the end of the experiment. Please call the experimenter.")

    win.close()
    core.quit()

# Run the experiment, specifying the main folder
main_folder = r"C:\ExperimentData\HRBP7\exp_package"  # Raw string ignores escape sequences
run_experiment(main_folder)
