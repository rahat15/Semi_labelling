# Semi-Auto Labelling Tool

## Introduction

We developed a semi-auto labelling tool that automatically labels images for a single class using a pretrained model.

### Important Features

1. The tool currently supports the YOLOv5 model as an input model in the form of a `.pt` weight file. We plan to add support for more models in the future.
2. Presently, only classification for one class is supported, but future versions will include support for multiple classes.
3. The output format is YOLO Darknet, though future versions may support other output formats as well.

## Installation

1. Open a terminal (PowerShell or any other) and navigate to the repository path.
2. Create a new environment using Conda or any environment manager (optional but recommended).
3. If using `pip`, run the following command:

   ```bash
   pip install -r requirements.txt
   ```

   If using Conda, create a new environment with the required packages using:

   ```bash
   conda env create -f environments.yml
   ```

4. Rename your model weights file (in `.pt` extension) to `model.pt` and place it in the `model` folder of the repository. (The default model included is for bees classification).
5. Input your images into the `input` folder of the repository.
6. Put the class name in `class_list.txt`. (Important!)
7. Run the following command in the terminal:

   ```bash
   python run.py -s IMAGE_SIZE -c CONFIDENCE_THRESHOLD
   ```
   You can specify the desired image size using the -s or --image_size argument followed by the size.
   *Note:* It takes only a single integer value as image size.

   Additionally, you can adjust the NMS confidence threshold using the -c or --conf_thres argument. For example, to set a specific confidence threshold for NMS, use:

   ```bash
   python run.py -s IMAGE_SIZE -c YOUR_CONFIDENCE_THRESHOLD
   ```

   Replace `YOUR_CONFIDENCE_THRESHOLD` with a decimal value between 0 and 1. This threshold will determine the minimum confidence required for a detection to be considered valid.
```
8. An interactive window will open as shown below:

   ![Example Image](https://github.com/scholar-2001/Semi_labelling/blob/master/Labellin_img.png?raw=true)

9. Perform the labeling operations using the following keybindings:

   - Right-click mouse: Remove a label
   - Left-click: Draw a new label
   - Q: Quit
   - E: Reveal edges
   - W/S: Navigate classes
   - A/D: Navigate images

10. After quitting, the labeled output will be available in YOLO Darknet format in the `output/Yolo_darknet` folder.
11. To clean up the output, you can run the following command (Make sure to save the output elsewhere before running this command!):

    ```bash
    python clean_up.py
    ```
## Docker Image

**Link**: [scholar2001/semi_labelling](https://hub.docker.com/r/scholar2001/semi_labelling)

### Running on Docker Desktop for Windows

1. **Pull the Image**:
   - Pull the Docker image to your local system using the following command:

     ```bash
     docker pull scholar2001/semi_labelling:latest
     ```

2. **Install VcXsrv Windows X Server**:
   - Before running the image, install VcXsrv Windows X Server from [here](https://sourceforge.net/projects/vcxsrv/). This X Server allows us to share an X11-session on a Windows host, enabling GUI support in Docker Containers on Windows.

3. **Run the Image**:
   - Execute the command below to run the image, mapping input and output folders, and setting the DISPLAY environment variable:

     ```bash
     docker run -it -v path/to/image/input/folder:/app/input -v path/to/output/folder:/app/output -e DISPLAY=host.docker.internal:0 -v /tmp/.X11-unix:/tmp/.X11-unix scholar2001/semi_labelling:latest
     ```

   Replace `path/to/image/input/folder` and `path/to/output/folder` with the respective paths on your local system.

### Running on Other Operating Systems

To use the Semi_Labelling tool on other operating systems, you need to forward the X11-Session to display the GUI.

Kindly refer to the documentation or instructions for your specific operating system on how to forward the X11-Session. This will enable you to run the tool with a GUI interface.

For further details and usage instructions, visit the Docker Hub repository: [scholar2001/semi_labelling](https://hub.docker.com/r/scholar2001/semi_labelling)

Note: The instructions provided here assume you have Docker installed and properly configured on your system. Ensure that you have the required permissions and access to the input and output folders on your local system.

## Credits

The code for `run.py` (which includes the GUI interface of the labeller) is based on this repository by [Cartucho](https://github.com/Cartucho): [https://github.com/Cartucho/OpenLabeling/blob/master/main/main.py](https://github.com/Cartucho/OpenLabeling/blob/master/main/main.py)