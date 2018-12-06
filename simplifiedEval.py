import glob
import json
import os
import shutil
import operator
import sys
import argparse

MINOVERLAP = 0.5 # default value (defined in the PASCAL VOC2012 challenge)

parser = argparse.ArgumentParser()
parser.add_argument('-q', '--quiet', help="minimalistic console output.", action="store_true")
parser.add_argument('-d','--dataset',dest='dataset',type=str)
parser.add_argument('-m','--modelName',dest='modelName',type=str)
args = parser.parse_args()


"""
 throw error and exit
"""
def error(msg):
  print(msg)
  sys.exit(0)

"""
 check if the number is a float between 0.0 and 1.0
"""
def is_float_between_0_and_1(value):
  try:
    val = float(value)
    if val > 0.0 and val < 1.0:
      return True
    else:
      return False
  except ValueError:
    return False

"""
 Calculate the AP given the recall and precision array
  1st) We compute a version of the measured precision/recall curve with
       precision monotonically decreasing
  2nd) We compute the AP as the area under this curve by numerical integration.
"""
def voc_ap(rec, prec):
  rec.insert(0, 0.0) # insert 0.0 at begining of list
  rec.append(1.0) # insert 1.0 at end of list
  mrec = rec[:]
  prec.insert(0, 0.0) # insert 0.0 at begining of list
  prec.append(0.0) # insert 0.0 at end of list
  mpre = prec[:]
  """
   This part makes the precision monotonically decreasing
    (goes from the end to the beginning)
  """
  for i in range(len(mpre)-2, -1, -1):
    mpre[i] = max(mpre[i], mpre[i+1])
  """
   This part creates a list of indexes where the recall changes
  """
  i_list = []
  for i in range(1, len(mrec)):
    if mrec[i] != mrec[i-1]:
      i_list.append(i) # if it was matlab would be i + 1
  """
   The Average Precision (AP) is the area under the curve
    (numerical integration)
  """
  ap = 0.0
  for i in i_list:
    ap += ((mrec[i]-mrec[i-1])*mpre[i])
  return ap, mrec, mpre


"""
 Convert the lines of a file to a list
"""
def file_lines_to_list(path):
  # open txt file lines to a list
  with open(path) as f:
    content = f.readlines()
  # remove whitespace characters like `\n` at the end of each line
  content = [x.strip() for x in content]
  return content


"""
 Create a "tmp_files/" and "results/" directory
"""
tmp_files_path = "tmp_files"
if not os.path.exists(tmp_files_path): # if it doesn't exist already
  os.makedirs(tmp_files_path)
results_files_path = "results"
if os.path.exists(results_files_path): # if it exist already
  # reset the results directory
  shutil.rmtree(results_files_path)

os.makedirs(results_files_path)

"""
 Ground-Truth
   Load each of the ground-truth files into a temporary ".json" file.
   Create a list of all the class names present in the ground-truth (gt_classes).
"""
# get a list with the ground-truth files
dataPath = '/root/data/data-'+args.dataset + '/'
print dataPath + 'ground-truth/set01'
ground_truth_files_list = glob.glob(dataPath+'ground-truth/set01/*.txt')
if len(ground_truth_files_list) == 0:
  error("Error: No ground-truth files found!")
ground_truth_files_list.sort()
# dictionary with counter per class
gt_counter_per_class = {}

targetPath = dataPath+'transformed/'+args.modelName+'/set01/'
print targetPath

for txt_file in ground_truth_files_list:
  #print(txt_file)
  file_id = txt_file.split(".txt",1)[0]
  file_id = os.path.basename(os.path.normpath(file_id))
  # check if there is a correspondent predicted objects file
  if not os.path.exists(targetPath+ file_id + ".txt"):
    print targetPath+file_id+".txt"
    error_msg = "Error. File not found: {}/".format(targetPath) +  file_id + ".txt\n"
    error_msg += "(You can avoid this error message by running extra/intersect-gt-and-pred.py)"
    error(error_msg)
  lines_list = file_lines_to_list(txt_file)
  # create ground-truth dictionary
  bounding_boxes = []
  for line in lines_list:
    if line.split(" ")[1] == "bbGt":
      continue
    try:
      class_name, left, top, right, bottom = line.split()
    except ValueError:
      error_msg = "Error: File " + txt_file + " in the wrong format.\n"
      error_msg += " Expected: <class_name> <left> <top> <right> <bottom>\n"
      error_msg += " Received: " + line
      error_msg += "\n\nIf you have a <class_name> with spaces between words you should remove them\n"
      error_msg += "by running the script \"rename_class.py\" in the \"extra/\" folder."
      error(error_msg)
    # check if class is in the ignore list, if yes skip
    if class_name in args.ignore:
      continue
    bbox = left + " " + top + " " + right + " " +bottom
    bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False})
    # count that object
    if class_name in gt_counter_per_class:
      gt_counter_per_class[class_name] += 1
    else:
      # if class didn't exist yet
      gt_counter_per_class[class_name] = 1
  # dump bounding_boxes into a ".json" file
  with open(tmp_files_path + "/" + file_id + "_ground_truth.json", 'w') as outfile:
    json.dump(bounding_boxes, outfile)

gt_classes = list(gt_counter_per_class.keys())
# let's sort the classes alphabetically
gt_classes = sorted(gt_classes)
n_classes = len(gt_classes)
#print(gt_classes)
#print(gt_counter_per_class)

"""
 Check format of the flag --set-class-iou (if used)
  e.g. check if class exists
"""
"""
 Predicted
   Load each of the predicted files into a temporary ".json" file.
"""
# get a list with the predicted files
predicted_files_list = glob.glob(targetPath+'*.txt')

predicted_files_list.sort()

for class_index, class_name in enumerate(gt_classes):
  bounding_boxes = []
  for txt_file in predicted_files_list:
    #print(txt_file)
    # the first time it checks if all the corresponding ground-truth files exist
    file_id = txt_file.split(".txt",1)[0]
    file_id = os.path.basename(os.path.normpath(file_id))
    if class_index == 0:
      if not os.path.exists(dataPath+'ground-truth/set01/' + file_id + ".txt"):
        error_msg = "Error. File not found: {}/ground-truth/set01/".format(dataPath) +  file_id + ".txt\n"
        error_msg += "(You can avoid this error message by running extra/intersect-gt-and-pred.py)"
        error(error_msg)
    lines = file_lines_to_list(txt_file)
    for line in lines:
      try:
        tmp_class_name, confidence, left, top, right, bottom = line.split()
      except ValueError:
        error_msg = "Error: File " + txt_file + " in the wrong format.\n"
        error_msg += " Expected: <class_name> <confidence> <left> <top> <right> <bottom>\n"
        error_msg += " Received: " + line
        error(error_msg)
      if tmp_class_name == class_name:
        #print("match")
        bbox = left + " " + top + " " + right + " " +bottom
        bounding_boxes.append({"confidence":confidence, "file_id":file_id, "bbox":bbox})
        #print(bounding_boxes)
  # sort predictions by decreasing confidence
  bounding_boxes.sort(key=lambda x:x['confidence'], reverse=True)
  with open(tmp_files_path + "/" + class_name + "_predictions.json", 'w') as outfile:
    json.dump(bounding_boxes, outfile)

"""
 Calculate the AP for each class
"""
sum_AP = 0.0
ap_dictionary = {}
# open file to store the results
with open(results_files_path + "/{}_{}_results.txt".format(args.dataset,args.modelName), 'w') as results_file:
  results_file.write("# AP and precision/recall per class\n")
  count_true_positives = {}
  for class_index, class_name in enumerate(gt_classes):
    count_true_positives[class_name] = 0
    """
     Load predictions of that class
    """
    predictions_file = tmp_files_path + "/" + class_name + "_predictions.json"
    predictions_data = json.load(open(predictions_file))

    """
     Assign predictions to ground truth objects
    """
    nd = len(predictions_data)
    tp = [0] * nd # creates an array of zeros of size nd
    fp = [0] * nd
    for idx, prediction in enumerate(predictions_data):
      file_id = prediction["file_id"]
      # assign prediction to ground truth object if any
      #   open ground-truth with that file_id
      gt_file = tmp_files_path + "/" + file_id + "_ground_truth.json"
      ground_truth_data = json.load(open(gt_file))
      ovmax = -1
      gt_match = -1
      # load prediction bounding-box
      bb = [ float(x) for x in prediction["bbox"].split() ]
      for obj in ground_truth_data:
        # look for a class_name match
        if obj["class_name"] == class_name:
          bbgt = [ float(x) for x in obj["bbox"].split() ]
          bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
          iw = bi[2] - bi[0] + 1
          ih = bi[3] - bi[1] + 1
          if iw > 0 and ih > 0:
            # compute overlap (IoU) = area of intersection / area of union
            ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                    + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
            ov = iw * ih / ua
            if ov > ovmax:
              ovmax = ov
              gt_match = obj

      # assign prediction as true positive or false positive
      # set minimum overlap
      min_overlap = MINOVERLAP
      if ovmax >= min_overlap:
        if not bool(gt_match["used"]):
          # true positive
          tp[idx] = 1
          gt_match["used"] = True
          count_true_positives[class_name] += 1
          # update the ".json" file
          with open(gt_file, 'w') as f:
              f.write(json.dumps(ground_truth_data))
          if show_animation:
            status = "MATCH!"
        else:
          # false positive (multiple detection)
          fp[idx] = 1
          if show_animation:
            status = "REPEATED MATCH!"
      else:
        # false positive
        fp[idx] = 1
        if ovmax > 0:
          status = "INSUFFICIENT OVERLAP"

      """
       Draw image to show animation
      """

    #print(tp)
    # compute precision/recall
    cumsum = 0
    for idx, val in enumerate(fp):
      fp[idx] += cumsum
      cumsum += val
    cumsum = 0
    for idx, val in enumerate(tp):
      tp[idx] += cumsum
      cumsum += val
    #print(tp)
    rec = tp[:]
    for idx, val in enumerate(tp):
      rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
    #print(rec)
    prec = tp[:]
    for idx, val in enumerate(tp):
      prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
    #print(prec)

    ap, mrec, mprec = voc_ap(rec, prec)
    sum_AP += ap
    text = "{0:.2f}%".format(ap*100) + " = " + class_name + " AP  " #class_name + " AP = {0:.2f}%".format(ap*100)
    """
     Write to results.txt
    """
    rounded_prec = [ '%.2f' % elem for elem in prec ]
    average_prec = sum(prec)/len(prec)
    rounded_rec = [ '%.2f' % elem for elem in rec ]
    average_rec = sum(rec)/len(rec)
    results_file.write(text + "\n Precision: " + str(average_prec) + "\n Recall   :" + str(average_rec) + "\n\n")
    if not args.quiet:
      print(text)
    ap_dictionary[class_name] = ap

    """
     Draw plot
    """
    if draw_plot:
      plt.plot(rec, prec, '-o')
      plt.fill_between(mrec, 0, mprec, alpha=0.2, edgecolor='r')
      # set window title
      fig = plt.gcf() # gcf - get current figure
      fig.canvas.set_window_title('AP ' + class_name)
      # set plot title
      plt.title('class: ' + text)
      #plt.suptitle('This is a somewhat long figure title', fontsize=16)
      # set axis titles
      plt.xlabel('Recall')
      plt.ylabel('Precision')
      # optional - set axes
      axes = plt.gca() # gca - get current axes
      axes.set_xlim([0.0,1.0])
      axes.set_ylim([0.0,1.05]) # .05 to give some extra space
      # Alternative option -> wait for button to be pressed
      #while not plt.waitforbuttonpress(): pass # wait for key display
      # Alternative option -> normal display
      #plt.show()
      # save the plot
      fig.savefig(results_files_path + "/classes/" + class_name + ".png")
      plt.cla() # clear axes for next plot

  if show_animation:
    cv2.destroyAllWindows()

  results_file.write("\n# mAP of all classes\n")
  mAP = sum_AP / n_classes
  text = "mAP = {0:.2f}%".format(mAP*100)
  results_file.write(text + "\n")
  print(text)

# remove the tmp_files directory
shutil.rmtree(tmp_files_path)

"""
 Count total of Predictions
"""
# iterate through all the files
pred_counter_per_class = {}
#all_classes_predicted_files = set([])
for txt_file in predicted_files_list:
  # get lines to list
  lines_list = file_lines_to_list(txt_file)
  for line in lines_list:
    class_name = line.split()[0]
    # check if class is in the ignore list, if yes skip
    if class_name in args.ignore:
      continue
    # count that object
    if class_name in pred_counter_per_class:
      pred_counter_per_class[class_name] += 1
    else:
      # if class didn't exist yet
      pred_counter_per_class[class_name] = 1
#print(pred_counter_per_class)
pred_classes = list(pred_counter_per_class.keys())


"""
 Write number of ground-truth objects per class to results.txt
"""
with open(results_files_path + "/{}_{}_results.txt".format(args.dataset,args.modelName), 'a') as results_file:
  results_file.write("\n# Number of ground-truth objects per class\n")
  for class_name in sorted(gt_counter_per_class):
    results_file.write(class_name + ": " + str(gt_counter_per_class[class_name]) + "\n")

"""
 Finish counting true positives
"""
for class_name in pred_classes:
  # if class exists in predictions but not in ground-truth then there are no true positives in that class
  if class_name not in gt_classes:
    count_true_positives[class_name] = 0
#print(count_true_positives)


"""
 Write number of predicted objects per class to results.txt
"""
with open(results_files_path + "/{}_{}_results.txt".format(args.dataset,args.modelName), 'a') as results_file:
  results_file.write("\n# Number of predicted objects per class\n")
  for class_name in sorted(pred_classes):
    n_pred = pred_counter_per_class[class_name]
    text = class_name + ": " + str(n_pred)
    text += " (tp:" + str(count_true_positives[class_name]) + ""
    text += ", fp:" + str(n_pred - count_true_positives[class_name]) + ")\n"
    results_file.write(text)

