import argparse
import os
import re
import cv2
import numpy as np


DELAY = 20 # keyboard delay (in milliseconds)
WITH_QT = False
try:
    cv2.namedWindow('Test')
    cv2.displayOverlay('Test', 'Test QT', 500)
    WITH_QT = True
except cv2.error:
    print('-> Please ignore this error message\n')
cv2.destroyAllWindows()


parser = argparse.ArgumentParser(description='Open-source image labeling tool')
parser.add_argument('-i', '--input_dir', default='input', type=str, help='Path to input directory')
parser.add_argument('-o', '--output_dir', default='output', type=str, help='Path to output directory')
parser.add_argument('-t', '--thickness', default='3', type=int, help='Bounding box and cross line thickness')
parser.add_argument('-s', '--image_size', default=640, type=int, help='Image size (single integer)')
parser.add_argument('-c', '--conf_thres', default=0.35, type=float, help='Confidence threshold')
parser.add_argument('-a', '--auto_annot', default='no', type=str, help='Choose whether to run auto annotator or not, Takes yes or no, default no')

args = parser.parse_args()
if(args.auto_annot == 'yes'):
    import auto_classify
    auto_classify.run_your_code(args.image_size,args.conf_thres)
class_index = 0
img_index = 0
img = None
img_objects = []

INPUT_DIR  = args.input_dir
OUTPUT_DIR = args.output_dir


WINDOW_NAME    = 'OpenLabeling'
TRACKBAR_IMG   = 'Image'
TRACKBAR_CLASS = 'Class'

annotation_formats = {'YOLO_darknet' : '.txt'}


prev_was_double_click = False
is_bbox_selected = False
selected_bbox = -1
LINE_THICKNESS = args.thickness

mouse_x = 0
mouse_y = 0
point_1 = (-1, -1)
point_2 = (-1, -1)

'''
    0,0 ------> x (width)
     |
     |  (Left,Top)
     |      *_________
     |      |         |
            |         |
     y      |_________|
  (height)            *
                (Right,Bottom)
'''


# Check if a point belongs to a rectangle
def pointInRect(pX, pY, rX_left, rY_top, rX_right, rY_bottom):
    return rX_left <= pX <= rX_right and rY_top <= pY <= rY_bottom



# Class to deal with bbox resizing
class dragBBox:
    '''
        LT -- MT -- RT
        |            |
        LM          RM
        |            |
        LB -- MB -- RB
    '''

    # Size of resizing anchors (depends on LINE_THICKNESS)
    sRA = LINE_THICKNESS * 2

    # Object being dragged
    selected_object = None

    # Flag indicating which resizing-anchor is dragged
    anchor_being_dragged = None

    '''
    \brief This method is used to check if a current mouse position is inside one of the resizing anchors of a bbox
    '''
    @staticmethod
    def check_point_inside_resizing_anchors(eX, eY, obj):
        _class_name, x_left, y_top, x_right, y_bottom = obj
        # first check if inside the bbox region (to avoid making 8 comparisons per object)
        if pointInRect(eX, eY,
                        x_left - dragBBox.sRA,
                        y_top - dragBBox.sRA,
                        x_right + dragBBox.sRA,
                        y_bottom + dragBBox.sRA):

            anchor_dict = get_anchors_rectangles(x_left, y_top, x_right, y_bottom)
            for anchor_key in anchor_dict:
                rX_left, rY_top, rX_right, rY_bottom = anchor_dict[anchor_key]
                if pointInRect(eX, eY, rX_left, rY_top, rX_right, rY_bottom):
                    dragBBox.anchor_being_dragged = anchor_key
                    break

    '''
    \brief This method is used to select an object if one presses a resizing anchor
    '''
    @staticmethod
    def handler_left_mouse_down(eX, eY, obj):
        dragBBox.check_point_inside_resizing_anchors(eX, eY, obj)
        if dragBBox.anchor_being_dragged is not None:
            dragBBox.selected_object = obj

    @staticmethod
    def handler_mouse_move(eX, eY):
        if dragBBox.selected_object is not None:
            class_name, x_left, y_top, x_right, y_bottom = dragBBox.selected_object

            # Do not allow the bbox to flip upside down (given a margin)
            margin = 3 * dragBBox.sRA
            change_was_made = False

            if dragBBox.anchor_being_dragged[0] == "L":
                # left anchors (LT, LM, LB)
                if eX < x_right - margin:
                    x_left = eX
                    change_was_made = True
            elif dragBBox.anchor_being_dragged[0] == "R":
                # right anchors (RT, RM, RB)
                if eX > x_left + margin:
                    x_right = eX
                    change_was_made = True

            if dragBBox.anchor_being_dragged[1] == "T":
                # top anchors (LT, RT, MT)
                if eY < y_bottom - margin:
                    y_top = eY
                    change_was_made = True
            elif dragBBox.anchor_being_dragged[1] == "B":
                # bottom anchors (LB, RB, MB)
                if eY > y_top + margin:
                    y_bottom = eY
                    change_was_made = True

            if change_was_made:
                action = "resize_bbox:{}:{}:{}:{}".format(x_left, y_top, x_right, y_bottom)
                edit_bbox(dragBBox.selected_object, action)
                # update the selected bbox
                dragBBox.selected_object = [class_name, x_left, y_top, x_right, y_bottom]

    '''
    \brief This method will reset this class
     '''
    @staticmethod
    def handler_left_mouse_up(eX, eY):
        if dragBBox.selected_object is not None:
            dragBBox.selected_object = None
            dragBBox.anchor_being_dragged = None

def display_text(text, time):
    if WITH_QT:
        cv2.displayOverlay(WINDOW_NAME, text, time)
    else:
        print(text)

def set_img_index(x):
    global img_index, img
    img_index = x
    img_path = IMAGE_PATH_LIST[img_index]
    img = cv2.imread(img_path)
    text = 'Showing image {}/{}, path: {}'.format(str(img_index), str(last_img_index), img_path)
    display_text(text, 1000)


def set_class_index(x):
    global class_index
    class_index = x
    text = 'Selected class {}/{} -> {}'.format(str(class_index), str(last_class_index), CLASS_LIST[class_index])
    display_text(text, 3000)


def draw_edges(tmp_img):
    blur = cv2.bilateralFilter(tmp_img, 3, 75, 75)
    edges = cv2.Canny(blur, 150, 250, 3)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    # Overlap image and edges together
    tmp_img = np.bitwise_or(tmp_img, edges)
    #tmp_img = cv2.addWeighted(tmp_img, 1 - edges_val, edges, edges_val, 0)
    return tmp_img


def decrease_index(current_index, last_index):
    current_index -= 1
    if current_index < 0:
        current_index = last_index
    return current_index


def increase_index(current_index, last_index):
    current_index += 1
    if current_index > last_index:
        current_index = 0
    return current_index


def draw_line(img, x, y, height, width, color):
    cv2.line(img, (x, 0), (x, height), color, LINE_THICKNESS)
    cv2.line(img, (0, y), (width, y), color, LINE_THICKNESS)


def yolo_format(class_index, point_1, point_2, width, height):
    # YOLO wants everything normalized
    # Order: class x_center y_center x_width y_height
    x_center = float((point_1[0] + point_2[0]) / (2.0 * width) )
    y_center = float((point_1[1] + point_2[1]) / (2.0 * height))
    x_width = float(abs(point_2[0] - point_1[0])) / width
    y_height = float(abs(point_2[1] - point_1[1])) / height
    items = map(str, [class_index, x_center, y_center, x_width, y_height])
    return ' '.join(items)



def findIndex(obj_to_find):
    #return [(ind, img_objects[ind].index(obj_to_find)) for ind in xrange(len(img_objects)) if item in img_objects[ind]]
    ind = -1

    ind_ = 0
    for listElem in img_objects:
        if listElem == obj_to_find:
            ind = ind_
            return ind
        ind_ = ind_+1

    return ind


def append_bb(ann_path, line, extension):

    with open(ann_path, 'a') as myfile:
        myfile.write(line + '\n')



def get_txt_object_data(obj, img_width, img_height):
    classId, centerX, centerY, bbox_width, bbox_height = obj.split()
    bbox_width = float(bbox_width)
    bbox_height  = float(bbox_height)
    centerX = float(centerX)
    centerY = float(centerY)

    class_index = int(classId)
    class_name = CLASS_LIST[class_index]
    xmin = int(img_width * centerX - img_width * bbox_width/2.0)
    xmax = int(img_width * centerX + img_width * bbox_width/2.0)
    ymin = int(img_height * centerY - img_height * bbox_height/2.0)
    ymax = int(img_height * centerY + img_height * bbox_height/2.0)
    return [class_name, class_index, xmin, ymin, xmax, ymax]


def get_anchors_rectangles(xmin, ymin, xmax, ymax):
    anchor_list = {}

    mid_x = (xmin + xmax) / 2
    mid_y = (ymin + ymax) / 2

    L_ = [xmin - dragBBox.sRA, xmin + dragBBox.sRA]
    M_ = [mid_x - dragBBox.sRA, mid_x + dragBBox.sRA]
    R_ = [xmax - dragBBox.sRA, xmax + dragBBox.sRA]
    _T = [ymin - dragBBox.sRA, ymin + dragBBox.sRA]
    _M = [mid_y - dragBBox.sRA, mid_y + dragBBox.sRA]
    _B = [ymax - dragBBox.sRA, ymax + dragBBox.sRA]

    anchor_list['LT'] = [L_[0], _T[0], L_[1], _T[1]]
    anchor_list['MT'] = [M_[0], _T[0], M_[1], _T[1]]
    anchor_list['RT'] = [R_[0], _T[0], R_[1], _T[1]]
    anchor_list['LM'] = [L_[0], _M[0], L_[1], _M[1]]
    anchor_list['RM'] = [R_[0], _M[0], R_[1], _M[1]]
    anchor_list['LB'] = [L_[0], _B[0], L_[1], _B[1]]
    anchor_list['MB'] = [M_[0], _B[0], M_[1], _B[1]]
    anchor_list['RB'] = [R_[0], _B[0], R_[1], _B[1]]

    return anchor_list


def draw_bbox_anchors(tmp_img, xmin, ymin, xmax, ymax, color):
    anchor_dict = get_anchors_rectangles(xmin, ymin, xmax, ymax)
    for anchor_key in anchor_dict:
        x1, y1, x2, y2 = anchor_dict[anchor_key]
        cv2.rectangle(tmp_img, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)
    return tmp_img

def draw_bboxes_from_file(tmp_img, annotation_paths, width, height):
    global img_objects#, is_bbox_selected, selected_bbox
    img_objects = []
    ann_path = None
    ann_path = next(path for path in annotation_paths if 'YOLO_darknet' in path)
    if os.path.isfile(ann_path):
        with open(ann_path) as fp:
            for idx, line in enumerate(fp):
                obj = line
                class_name, class_index, xmin, ymin, xmax, ymax = get_txt_object_data(obj, width, height)
                #print('{} {} {} {} {}'.format(class_index, xmin, ymin, xmax, ymax))
                img_objects.append([class_index, xmin, ymin, xmax, ymax])
                color = class_rgb[class_index].tolist()
                # draw bbox
                cv2.rectangle(tmp_img, (xmin, ymin), (xmax, ymax), color, LINE_THICKNESS)
                # draw resizing anchors if the object is selected
                if is_bbox_selected:
                    if idx == selected_bbox:
                        tmp_img = draw_bbox_anchors(tmp_img, xmin, ymin, xmax, ymax, color)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(tmp_img, class_name, (xmin, ymin - 5), font, 1.5, color, LINE_THICKNESS, cv2.LINE_AA)
    return tmp_img


def get_bbox_area(x1, y1, x2, y2):
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    return width*height


def set_selected_bbox(set_class):
    global is_bbox_selected, selected_bbox
    smallest_area = -1
    # if clicked inside multiple bboxes selects the smallest one
    for idx, obj in enumerate(img_objects):
        ind, x1, y1, x2, y2 = obj
        x1 = x1 - dragBBox.sRA
        y1 = y1 - dragBBox.sRA
        x2 = x2 + dragBBox.sRA
        y2 = y2 + dragBBox.sRA
        if pointInRect(mouse_x, mouse_y, x1, y1, x2, y2):
            is_bbox_selected = True
            tmp_area = get_bbox_area(x1, y1, x2, y2)
            if tmp_area < smallest_area or smallest_area == -1:
                smallest_area = tmp_area
                selected_bbox = idx
                if set_class:
                    # set class to the one of the selected bounding box
                    cv2.setTrackbarPos(TRACKBAR_CLASS, WINDOW_NAME, ind)


def is_mouse_inside_delete_button():
    for idx, obj in enumerate(img_objects):
        if idx == selected_bbox:
            _ind, x1, y1, x2, y2 = obj
            x1_c, y1_c, x2_c, y2_c = get_close_icon(x1, y1, x2, y2)
            if pointInRect(mouse_x, mouse_y, x1_c, y1_c, x2_c, y2_c):
                return True
    return False


def edit_bbox(obj_to_edit, action):
    ''' action = `delete`
                 `change_class:new_class_index`
                 `resize_bbox:new_x_left:new_y_top:new_x_right:new_y_bottom`
    '''
    if 'change_class' in action:
        new_class_index = int(action.split(':')[1])
    elif 'resize_bbox' in action:
        new_x_left = max(0, int(action.split(':')[1]))
        new_y_top = max(0, int(action.split(':')[2]))
        new_x_right = min(width, int(action.split(':')[3]))
        new_y_bottom = min(height, int(action.split(':')[4]))

    # 1. initialize bboxes_to_edit_dict
    #    (we use a dict since a single label can be associated with multiple ones in videos)
    bboxes_to_edit_dict = {}
    current_img_path = IMAGE_PATH_LIST[img_index]
    bboxes_to_edit_dict[current_img_path] = obj_to_edit

    # 2. loop through bboxes_to_edit_dict and edit the corresponding annotation files
    for path in bboxes_to_edit_dict:
        obj_to_edit = bboxes_to_edit_dict[path]
        class_index, xmin, ymin, xmax, ymax = map(int, obj_to_edit)

        for ann_path in get_annotation_paths(path, annotation_formats):
            # edit YOLO file
            with open(ann_path, 'r') as old_file:
                lines = old_file.readlines()

            yolo_line = yolo_format(class_index, (xmin, ymin), (xmax, ymax), width, height) # TODO: height and width ought to be stored
            ind = findIndex(obj_to_edit)
            i=0

            with open(ann_path, 'w') as new_file:
                for line in lines:

                    if i != ind:
                        new_file.write(line)

                    elif 'change_class' in action:
                        new_yolo_line = yolo_format(new_class_index, (xmin, ymin), (xmax, ymax), width, height)
                        new_file.write(new_yolo_line + '\n')
                    elif 'resize_bbox' in action:
                        new_yolo_line = yolo_format(class_index, (new_x_left, new_y_top), (new_x_right, new_y_bottom), width, height)
                        new_file.write(new_yolo_line + '\n')

                    i=i+1




def mouse_listener(event, x, y, flags, param):
    # mouse callback function
    global is_bbox_selected, prev_was_double_click, mouse_x, mouse_y, point_1, point_2

    set_class = True
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x = x
        mouse_y = y
    elif event == cv2.EVENT_LBUTTONDBLCLK:
        prev_was_double_click = True
        #print('Double click')
        point_1 = (-1, -1)
        # if clicked inside a bounding box we set that bbox
        set_selected_bbox(set_class)
    # By AlexeyGy: delete via right-click
    elif event == cv2.EVENT_RBUTTONDOWN:
        set_class = False
        set_selected_bbox(set_class)
        if is_bbox_selected:
            obj_to_edit = img_objects[selected_bbox]
            edit_bbox(obj_to_edit, 'delete')
            is_bbox_selected = False
    elif event == cv2.EVENT_LBUTTONDOWN:
        if prev_was_double_click:
            #print('Finish double click')
            prev_was_double_click = False
        else:
            #print('Normal left click')

            # Check if mouse inside on of resizing anchors of the selected bbox
            if is_bbox_selected:
                dragBBox.handler_left_mouse_down(x, y, img_objects[selected_bbox])

            if dragBBox.anchor_being_dragged is None:
                if point_1[0] == -1:
                    if is_bbox_selected:
                        if is_mouse_inside_delete_button():
                            set_selected_bbox(set_class)
                            obj_to_edit = img_objects[selected_bbox]
                            edit_bbox(obj_to_edit, 'delete')
                        is_bbox_selected = False
                    else:
                        # first click (start drawing a bounding box or delete an item)

                        point_1 = (x, y)
                else:
                    # minimal size for bounding box to avoid errors
                    threshold = 5
                    if abs(x - point_1[0]) > threshold or abs(y - point_1[1]) > threshold:
                        # second click
                        point_2 = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        if dragBBox.anchor_being_dragged is not None:
            dragBBox.handler_left_mouse_up(x, y)



def get_close_icon(x1, y1, x2, y2):
    percentage = 0.05
    height = -1
    while height < 15 and percentage < 1.0:
        height = int((y2 - y1) * percentage)
        percentage += 0.1
    return (x2 - height), y1, x2, (y1 + height)


def draw_close_icon(tmp_img, x1_c, y1_c, x2_c, y2_c):
    red = (0,0,255)
    cv2.rectangle(tmp_img, (x1_c + 1, y1_c - 1), (x2_c, y2_c), red, -1)
    white = (255, 255, 255)
    cv2.line(tmp_img, (x1_c, y1_c), (x2_c, y2_c), white, 2)
    cv2.line(tmp_img, (x1_c, y2_c), (x2_c, y1_c), white, 2)
    return tmp_img


def draw_info_bb_selected(tmp_img):
    for idx, obj in enumerate(img_objects):
        ind, x1, y1, x2, y2 = obj
        if idx == selected_bbox:
            x1_c, y1_c, x2_c, y2_c = get_close_icon(x1, y1, x2, y2)
            draw_close_icon(tmp_img, x1_c, y1_c, x2_c, y2_c)
    return tmp_img


def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in _nsre.split(s)]


def nonblank_lines(f):
    for l in f:
        line = l.rstrip()
        if line:
            yield line


def get_annotation_paths(img_path, annotation_formats):
    annotation_paths = []
    for ann_dir, ann_ext in annotation_formats.items():
        new_path = os.path.join(OUTPUT_DIR, ann_dir)
        new_path = os.path.join(new_path, os.path.basename(os.path.normpath(img_path))) #img_path.replace(INPUT_DIR, new_path, 1)
        pre_path, img_ext = os.path.splitext(new_path)
        new_path = new_path.replace(img_ext, ann_ext, 1)
        annotation_paths.append(new_path)
    return annotation_paths



def save_bounding_box(annotation_paths, class_index, point_1, point_2, width, height):
    for ann_path in annotation_paths:
        line = yolo_format(class_index, point_1, point_2, width, height)
        append_bb(ann_path, line, '.txt')



def complement_bgr(color):
    lo = min(color)
    hi = max(color)
    k = lo + hi
    return tuple(k - u for u in color)

# change to the directory of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    # load all images and videos (with multiple extensions) from a directory using OpenCV
    IMAGE_PATH_LIST = []
    for f in sorted(os.listdir(INPUT_DIR), key = natural_sort_key):
        f_path = os.path.join(INPUT_DIR, f)
        if os.path.isdir(f_path):
            # skip directories
            continue
        IMAGE_PATH_LIST.append(f_path)
    last_img_index = len(IMAGE_PATH_LIST) - 1
    for ann_dir in annotation_formats:
        new_dir = os.path.join(OUTPUT_DIR, ann_dir)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)

    # create empty annotation files for each image, if it doesn't exist already
    for img_path in IMAGE_PATH_LIST:
        # image info for the .xml file
        test_img = cv2.imread(img_path)
        abs_path = os.path.abspath(img_path)
        folder_name = os.path.dirname(img_path)
        image_name = os.path.basename(img_path)
        img_height, img_width, depth = (str(number) for number in test_img.shape)

        for ann_path in get_annotation_paths(img_path, annotation_formats):
            if not os.path.isfile(ann_path):
                open(ann_path, 'a').close()

    # load class list
    with open('class_list.txt') as f:
        CLASS_LIST = list(nonblank_lines(f))
    #print(CLASS_LIST)
    last_class_index = len(CLASS_LIST) - 1

    # Make the class colors the same each session
    # The colors are in BGR order because we're using OpenCV
    class_rgb = [
        (0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 255, 255),
        (255, 0, 255), (192, 192, 192), (128, 128, 128), (128, 0, 0),
        (128, 128, 0), (0, 128, 0), (128, 0, 128), (0, 128, 128), (0, 0, 128)]
    class_rgb = np.array(class_rgb)
    # If there are still more classes, add new colors randomly
    num_colors_missing = len(CLASS_LIST) - len(class_rgb)
    if num_colors_missing > 0:
        more_colors = np.random.randint(0, 255+1, size=(num_colors_missing, 3))
        class_rgb = np.vstack([class_rgb, more_colors])

    # create window
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WINDOW_NAME, 1000, 700)
    cv2.setMouseCallback(WINDOW_NAME, mouse_listener)

    # selected image
    cv2.createTrackbar(TRACKBAR_IMG, WINDOW_NAME, 0, last_img_index, set_img_index)

    # selected class
    if last_class_index != 0:
        cv2.createTrackbar(TRACKBAR_CLASS, WINDOW_NAME, 0, last_class_index, set_class_index)

    # initialize
    set_img_index(0)
    edges_on = False

    display_text('Welcome!\n Press [h] for help.', 4000)

    # loop
    while True:
        color = class_rgb[class_index].tolist()
        # clone the img
        tmp_img = img.copy()
        height, width = tmp_img.shape[:2]
        if edges_on == True:
            # draw edges
            tmp_img = draw_edges(tmp_img)
        # draw vertical and horizontal guide lines
        draw_line(tmp_img, mouse_x, mouse_y, height, width, color)
        # write selected class
        class_name = CLASS_LIST[class_index]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        margin = 10
        text_width, text_height = cv2.getTextSize(class_name, font, font_scale, LINE_THICKNESS)[0]
        tmp_img = cv2.rectangle(tmp_img, (mouse_x + LINE_THICKNESS, mouse_y - LINE_THICKNESS), (mouse_x + text_width + margin, mouse_y - text_height - margin), complement_bgr(color), -1)
        tmp_img = cv2.putText(tmp_img, class_name, (mouse_x + margin, mouse_y - margin), font, font_scale, color, LINE_THICKNESS, cv2.LINE_AA)
        # get annotation paths
        img_path = IMAGE_PATH_LIST[img_index]
        annotation_paths = get_annotation_paths(img_path, annotation_formats)
        if dragBBox.anchor_being_dragged is not None:
            dragBBox.handler_mouse_move(mouse_x, mouse_y)
        # draw already done bounding boxes
        tmp_img = draw_bboxes_from_file(tmp_img, annotation_paths, width, height)
        # if bounding box is selected add extra info
        if is_bbox_selected:
            tmp_img = draw_info_bb_selected(tmp_img)
        # if first click
        if point_1[0] != -1:
            # draw partial bbox
            cv2.rectangle(tmp_img, point_1, (mouse_x, mouse_y), color, LINE_THICKNESS)
            # if second click
            if point_2[0] != -1:
                # save the bounding box
                save_bounding_box(annotation_paths, class_index, point_1, point_2, width, height)
                # reset the points
                point_1 = (-1, -1)
                point_2 = (-1, -1)

        cv2.imshow(WINDOW_NAME, tmp_img)
        pressed_key = cv2.waitKey(DELAY)

        if dragBBox.anchor_being_dragged is None:
            ''' Key Listeners START '''
            if pressed_key == ord('a') or pressed_key == ord('d'):
                # show previous image key listener
                if pressed_key == ord('a'):
                    img_index = decrease_index(img_index, last_img_index)
                # show next image key listener
                elif pressed_key == ord('d'):
                    img_index = increase_index(img_index, last_img_index)
                set_img_index(img_index)
                cv2.setTrackbarPos(TRACKBAR_IMG, WINDOW_NAME, img_index)
            elif pressed_key == ord('s') or pressed_key == ord('w'):
                # change down current class key listener
                if pressed_key == ord('s'):
                    class_index = decrease_index(class_index, last_class_index)
                # change up current class key listener
                elif pressed_key == ord('w'):
                    class_index = increase_index(class_index, last_class_index)
                draw_line(tmp_img, mouse_x, mouse_y, height, width, color)
                set_class_index(class_index)
                cv2.setTrackbarPos(TRACKBAR_CLASS, WINDOW_NAME, class_index)
                if is_bbox_selected:
                    obj_to_edit = img_objects[selected_bbox]
                    edit_bbox(obj_to_edit, 'change_class:{}'.format(class_index))
            # help key listener
            elif pressed_key == ord('h'):
                text = ('[e] to show edges;\n'
                        '[q] to quit;\n'
                        '[a] or [d] to change Image;\n'
                        '[w] or [s] to change Class.\n'
                        )
                display_text(text, 5000)
            # show edges key listener
            elif pressed_key == ord('e'):
                if edges_on == True:
                    edges_on = False
                    display_text('Edges turned OFF!', 1000)
                else:
                    edges_on = True
                    display_text('Edges turned ON!', 1000)
            # quit key listener
            elif pressed_key == ord('q'):
                break
            ''' Key Listeners END '''

        if WITH_QT:
            # if window gets closed then quit
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

    cv2.destroyAllWindows()
