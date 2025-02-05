import os
import json
import sys
import numpy as np
import matplotlib.pyplot as plt

# ATEAM_LABELS = ['legend_label', 'chart_title', 'tick_label', 'axis_title', 'other', 'legend_title']
# SYNTH_LABELS = ['legend_label', 'chart_title', 'tick_label', 'axis_title']


def get_confusion_matrix(confusion, unique_labels):
    label_idx_map = {label : i for i, label in enumerate(unique_labels)}
    idx_label_map = {i : label for label, i in label_idx_map.items()}
    cmat = np.zeros((len(label_idx_map), len(label_idx_map)))
    for ID, pair in confusion.items():
        truth, pred = pair
        if pred is None or pred not in label_idx_map:
            continue
        if truth not in label_idx_map:
            continue
        t = label_idx_map[truth]
        p = label_idx_map[pred]
        cmat[t, p] += 1
    norm = cmat.sum(axis=1).reshape(-1, 1)
    cmat /= norm
    return cmat, idx_label_map


def plot_confusion_matrix(cm, classes, output_img_path):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    # fig.tight_layout()
    fig.savefig(output_img_path, bbox_inches='tight')
    plt.show()

def eval_task3(gt_folder, result_folder, output_img_path, classes_ignore):
    gt_label_map = {}
    result_label_map = {}
    metrics = {}
    confusion = {}
    ignore = set()

    # get list of files ...
    result_files = os.listdir(result_folder)
    gt_files = os.listdir(gt_folder)

    # for all GT files ... 
    for gt_file in gt_files:
        # read and get the roles ... 
        gt_id = '.'.join(gt_file.split('.')[:-1])
        with open(os.path.join(gt_folder, gt_file), 'r') as f:
            gt = json.load(f)    
        text_roles = gt['task3']['output']['text_roles']

        for text_role in text_roles:
            text_id = text_role['id']
            role = text_role['role'].lower().strip()

            role_id = '{}__sep__{}'.format(gt_id, text_id)
            
            # SOME LABELS IN PMC NOT PRESENT IN SYNTHETIC, TO BE CONSIDERED AS DONT CARE FOR EVAL
            if role in classes_ignore:
                ignore.add(role_id)
                continue

            if role in gt_label_map:
                gt_label_map[role].append(role_id)
            else:
                gt_label_map[role] = [role_id]

            confusion[role_id] = [role, None]

    if len(ignore) > 0:
        print("A total of {0:d} entries will be ignored".format(len(ignore)))
    else:
        print("No entries will be ignored")

    unique_roles = set()
    for result_file in result_files:
        result_id = '.'.join(result_file.split('.')[:-1])
        with open(os.path.join(result_folder, result_file), 'r') as f:
            result = json.load(f)
        try:
            if "task1.3" in result:
                result['task3'] = result["task1.3"]
            
            if 'text_roles' in result['task3']['output']:
                text_roles = result['task3']['output']['text_roles']
                # this is due to wrong json format in a submission
            else:
                text_roles = result['task3']['output']['text_blocks']

            for text_role in text_roles:
                text_id = text_role['id']
                role = text_role['role'].lower().strip()

                role_id = '{}__sep__{}'.format(result_id, text_id)

                # SOME LABELS IN PMC NOT PRESENT IN SYNTHETIC, TO BE CONSIDERED AS DONT CARE FOR EVAL
                unique_roles.add(role)
                if role_id in ignore:
                    continue

                if role in result_label_map:
                    result_label_map[role].append(role_id)
                else:
                    result_label_map[role] = [role_id]

                confusion[role_id][1] = role
        except Exception as e:
            print(e)
            print('invalid result json format in {} please check against provided samples'.format(result_file))
            continue

    # compute the metrics ...
    total_recall = 0.0
    total_precision = 0.0
    total_fmeasure = 0.0

    # print(unique_roles)
    for label, gt_instances in gt_label_map.items():
        if label in result_label_map:
            res_instances = set(result_label_map[label])
        else:
            res_instances = {}
        gt_instances = set(gt_instances)
        intersection = gt_instances.intersection(res_instances)
        print(label, len(gt_instances), len(res_instances), len(intersection))

        recall = len(intersection) / float(len(gt_instances))

        if len(intersection) == 0 or len(res_instances) == 0:
            precision = 0
        else:
            precision = len(intersection) / float(len(res_instances))

        if recall == 0 and precision == 0:
            f_measure = 0.0
        else:
            f_measure = 2 * recall * precision / (recall + precision)

        total_recall += recall
        total_precision += precision
        total_fmeasure += f_measure
        metrics[label] = (recall, precision, f_measure)

        print('Recall for class {}: {}'.format(label, recall))
        print('Precision for class {}: {}'.format(label, precision))
        print('F-measure for class {}: {}'.format(label, f_measure))

    total_recall /= len(gt_label_map)
    total_precision /= len(gt_label_map)
    total_fmeasure /= len(gt_label_map)

    print('Average Recall across {} classes: {}'.format(len(gt_label_map), total_recall))
    print('Average Precision across {} classes: {}'.format(len(gt_label_map), total_precision))
    print('Average F-Measure across {} classes: {}'.format(len(gt_label_map), total_fmeasure))

    print('Computing Confusion Matrix')
    classes = sorted(list(gt_label_map.keys()))
    cm, idx_label_map = get_confusion_matrix(confusion, classes)
    plot_confusion_matrix(cm, classes, output_img_path)


def main():
    if len(sys.argv) < 4:
        print("Usage guide: ")
        print('\tpython metric3.py <ground_truth_folder> <result_folder> <confusion_matrix_path> [<ignore>]')
        return

    gt_dir = sys.argv[1]
    results_dir = sys.argv[2]
    conf_matrix_path = sys.argv[3]

    if len(sys.argv) >= 5:
        to_ignore_filename = sys.argv[4]
        print("Will ignored the classes listed in: " + to_ignore_filename)

        with open(to_ignore_filename, "r") as in_file:
            all_ignore_lines = in_file.readlines()

        classes_ignore = [line.strip() for line in all_ignore_lines]
    else:
        print("No classes in Ground Truth will be ignored")
        classes_ignore = []

    eval_task3(gt_dir, results_dir, conf_matrix_path, classes_ignore)
    """
    try:
        eval_task3(gt_dir, results_dir, conf_matrix_path)
    except Exception as e:
        print(e)
    """

if __name__ == '__main__':
    main()    