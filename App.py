from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import cv2
import numpy as np
from keras.models import load_model
from collections import deque
import matplotlib.pyplot as plt

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'mp4'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
IMG_SIZE = 128
ColorChannels = 3

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def delete_images_from_folder(file_extensions=[".jpg", ".png", ".jpeg"]):
    RESULTS_FOLDER 
    # List all files from the directory
    files = os.listdir(RESULTS_FOLDER)

    # Filter the list by the specified extensions
    images = [f for f in files if any(f.endswith(extension) for extension in file_extensions)]

    # Delete each image
    for image in images:
        os.remove(os.path.join(RESULTS_FOLDER, image))
    print(f"Deleted {len(images)} images from {RESULTS_FOLDER}")
    
def print_results(video_path, limit=None):
    # if not os.path.exists(app.config['RESULTS_FOLDER']):
    #     os.makedirs(app.config['RESULTS_FOLDER'])

    # # [Your model loading and preprocessing code here]

    # # Assuming frames are saved in the RESULTS_FOLDER
    # # This is just a placeholder; replace with your own logic.
    # vs = cv2.VideoCapture(video_path)
    # count = 0
    # while True:
    #     ret, frame = vs.read()
    #     if not ret:
    #         break
    #     save_path = os.path.join(app.config['RESULTS_FOLDER'], f"frame_{count}.jpg")
    #     cv2.imwrite(save_path, frame)
    #     count += 1
    fig=plt.figure(figsize=(16, 30))
    if not os.path.exists('output'):
        os.mkdir('output')

    print("Loading model ...")
    model = load_model('./Mobinet.h5')
    Q = deque(maxlen=128)
    
    # Create a results directory inside static to save frames


    vs = cv2.VideoCapture(video_path)
    writer = None
    (W, H) = (None, None)
    count = 0
    while True:
            (grabbed, frame) = vs.read()
            ID = vs.get(1)
            if not grabbed:
                break
            try:
                if (ID % 7 == 0):
                    count = count + 1
                    n_frames = len(frame)

                    if W is None or H is None:
                        (H, W) = frame.shape[:2]

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    output = cv2.resize(frame, (512, 360)).copy()
                    frame = cv2.resize(frame, (128, 128)).astype("float32")
                    frame = frame.reshape(IMG_SIZE, IMG_SIZE, 3) / 255
                    preds = model.predict(np.expand_dims(frame, axis=0))[0]
                    Q.append(preds)

                    results = np.array(Q).mean(axis=0)
                    i = (preds > 0.56)[0] #np.argmax(results)

                    label = i

                    text = "Violence: {}".format(label)
                    print('prediction:', text)
                    file = open("output.txt",'w')
                    file.write(text)
                    file.close()
                    color = (0, 255, 0)

                    if label:
                        color = (255, 0, 0)
                    else:
                        color = (0, 255, 0)

                    cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, color, 3)
                    
                    
                    
                                    # Save the image frame to results directory
                    image_name = os.path.join(RESULTS_FOLDER, f"frame_{count}.jpg")
                    plt.imshow(output)
                    plt.savefig(image_name)
                    plt.close() 


                    # saving mp4 with labels but cv2.imshow is not working with this notebook
                    # if writer is None:
                    #         fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                    #         writer = cv2.VideoWriter("output.mp4", fourcc, 60,
                    #                 (W, H), True)

                    # writer.write(output)
                    # #cv2.imshow("Output", output)
                    # image_name = os.path.join(RESULTS_FOLDER, f"frame_{count}.jpg")
                    # fig.add_subplot(8, 3, count)
                    # plt.imshow(output)
                    # plt.savefig('foo.png')

                if limit and count > limit:
                    break
            except:
                print('Error')
                    

@app.route('/', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            delete_images_from_folder()
            print_results(filename)
            return redirect(url_for('show_results'))
    return render_template('upload.html')

@app.route('/results')
def show_results():
    # image_names = os.listdir(app.config['RESULTS_FOLDER'])
    # return render_template('results.html', image_names=image_names)
    image_folder = os.path.join('static', 'results')
    image_names = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    image_paths = [os.path.join(image_folder, image_name) for image_name in image_names]
    return render_template('results.html', image_paths=image_paths)

if __name__ == '__main__':
    app.run(debug=True)
