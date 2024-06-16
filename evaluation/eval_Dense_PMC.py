import os 
import json 
import re
import random 

pred_dir = '/home/csgrad/sahmed9/reps/RealCQA/code/eval_result/dense_chart_148000/' 

json_dir = '/home/csgrad/sahmed9/reps/chartinfo-cqa/dataset/all_test_train_22/jsons'


def superscript_to_int(superscript_str):
    """Convert superscript string to integer."""
    superscript_map = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", 
                       "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"}
    return int("".join(superscript_map.get(char, '') for char in superscript_str))

def convert_to_numeric_or_string(value):
    # Patterns to match
    patterns = [
        (r'^(\d+)$', lambda x: int(x)),  # Integer
        (r'^(\d+\.\d+)$', lambda x: float(x)),  # Float
        (r'^(\d+(\.\d+)?[eE][+-]?\d+)$', lambda x: float(x)),  # Scientific notation
        (r'^(\d+(\.\d+)?)[×x*]10\^([+-]?\d+)$', lambda x: float(x.replace('×', 'e').replace('x', 'e').replace('*', 'e').replace('^', ''))),  # Custom scientific notation with "x" or "*"
        (r'^(\d{1,3}(,\d{3})*(\.\d+)?)$', lambda x: float(x.replace(',', ''))),  # Float with commas
        # Superscript scientific notation (e.g., "2.5×10⁵")
        (r'^(\d+(\.\d+)?)[×x*]10([⁰¹²³⁴⁵⁶⁷⁸⁹]+)$', lambda x: float(x.split('×')[0] + 'e' + str(superscript_to_int(x.split('×')[1][2:])))),
    ]
    
    for pattern, converter in patterns:
        match = re.match(pattern, value)
        if match:
            try:
                return converter(match.group(0))
            except ValueError:
                pass  # If conversion fails, try the next pattern
    
    # Fallback: return the original value as string if no pattern matches
    return value

def analyze_array(values):
    all_numeric = all(isinstance(value, (int, float)) for value in values)
    if all_numeric:
        return (min(values), max(values))
    else:
        return 'categorical'
    
def extract_txt_getRect(tb, filtered_ids ):
    results = []
    for block in tb:
        if block["id"] in filtered_ids:
            text = block["text"]
            polygon = block["polygon"]
            x = polygon["x0"]
            y = polygon["y0"]
            w = polygon["x1"] - polygon["x0"]  # Assuming x1 and x0 are the horizontal bounds
            h = polygon["y2"] - polygon["y0"]  # Assuming y2 and y0 are the vertical bounds
            results.append({
                "id": block["id"],
                "text": text,
                "bounding_box": {"x": x, "y": y, "width": w, "height": h}
            })
    return results

def get_combined_centroid(tb, filtered_ids):
    sum_x = 0
    sum_y = 0

    for block in tb:
        if block["id"] in filtered_ids:
            polygon = block["polygon"]
            centroid_x = (polygon["x0"] + polygon["x1"] + polygon["x2"] + polygon["x3"]) / 4
            centroid_y = (polygon["y0"] + polygon["y1"] + polygon["y2"] + polygon["y3"]) / 4
            sum_x += centroid_x
            sum_y += centroid_y
    
    filtered_count = len(filtered_ids)
    return (sum_x / filtered_count, sum_y / filtered_count)

def distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

def assign_axis_title(xax_ctr, yax_ctr, axs_titles, plot_bb):
    assignments = {'x_title': None, 'y_title': None}
    # Case when axis centroids are not provided
    if not xax_ctr or not yax_ctr:
        # Determine top-left-most title to assign as y_title
        titles_sorted_by_position = sorted(axs_titles, key=lambda t: (t["bounding_box"]["y"], t["bounding_box"]["x"]))
        if len(titles_sorted_by_position) > 1:
            assignments['y_title'] = titles_sorted_by_position[0]["text"]
            assignments['x_title'] = titles_sorted_by_position[1]["text"]
        elif titles_sorted_by_position:
            # When there's only one title, use plot_bb to decide
            title = titles_sorted_by_position[0]
            title_centroid = ((title["bounding_box"]["x"] + title["bounding_box"]["width"] / 2),
                              (title["bounding_box"]["y"] + title["bounding_box"]["height"] / 2))
            plot_centroid = (plot_bb["x0"] + plot_bb["width"] / 2, plot_bb["y0"] + plot_bb["height"] / 2)
            # If title is above or to the left of plot centroid, it's y_title, else x_title
            if title_centroid[0] < plot_centroid[0] or title_centroid[1] < plot_centroid[1]:
                assignments['y_title'] = title["text"]
            else:
                assignments['x_title'] = title["text"]
    else:
        # Original logic for assigning axis titles based on proximity
        for title in axs_titles:
            title_centroid = ((title["bounding_box"]["x"] + title["bounding_box"]["width"] / 2),
                              (title["bounding_box"]["y"] + title["bounding_box"]["height"] / 2))
            dist_to_x = distance(title_centroid, xax_ctr) if xax_ctr else float('inf')
            dist_to_y = distance(title_centroid, yax_ctr) if yax_ctr else float('inf')
            
            if dist_to_x < dist_to_y:
                closest_axis = 'x_title'
            else:
                closest_axis = 'y_title'
            assignments[closest_axis] = title["text"]
    return assignments




def check1(js_obj):
    if 'task1' in js_obj and js_obj['task1'] is not None and \
               'output' in js_obj['task1'] and js_obj['task1']['output'] is not None and \
               'chart_type' in js_obj['task1']['output'] and \
               js_obj['task1']['output']['chart_type'] is not None:
        return True 
    return False

def check2(js_obj):
    if 'task2' in js_obj and js_obj['task2'] is not None and \
               'output' in js_obj['task2'] and js_obj['task2']['output'] is not None and \
               'text_blocks' in js_obj['task2']['output'] and \
               len(js_obj['task2']['output']['text_blocks']) > 0:
        return True 
    return False

def check3(js_obj):
    if 'task3' in js_obj and js_obj['task3'] is not None and \
               'output' in js_obj['task3'] and js_obj['task3']['output'] is not None and \
               'text_roles' in js_obj['task3']['output'] and \
               len(js_obj['task3']['output']['text_roles']) > 0:
        return True 
    return False

def check4(js_obj):
    if 'task4' in js_obj and js_obj['task4'] is not None and \
               'output' in js_obj['task4'] and js_obj['task4']['output'] is not None and \
               'axes' in js_obj['task4']['output'] and \
               (len(js_obj['task4']['output']['axes']['x-axis']) > 0 or\
                len(js_obj['task4']['output']['axes']['y-axis'])) > 0:
        return True 
    return False

def check5(js_obj):
    if 'task5' in js_obj and js_obj['task5'] is not None and \
               'output' in js_obj['task5'] and js_obj['task5']['output'] is not None and \
               'legend_pairs' in js_obj['task5']['output'] and \
               len(js_obj['task5']['output']['legend_pairs']) > 0:
        return True 
    return False

def check6(js_obj):
    if 'task6' in js_obj and js_obj['task6'] is not None and \
               'output' in js_obj['task6'] and js_obj['task6']['output'] is not None and \
               'data series' in js_obj['task6']['output'] and \
               len(js_obj['task6']['output']['data series']) > 0:
        return True 
    return False

def extract_chart_variables(js_obj):
    chart_variables = {
        'chart_type':    None,
        'all_txt':       None,
        'xmin':          None, 
        'xmax':          None, 
        'ymin':          None, 
        'ymax':          None,
        'x_title':       None, 
        'y_title':       None,
        'x_ticks':       None,
        'y_ticks':       None, 
        'categorical':   None,
        'number_of_ds':  None, 
        'legend_labels': None 
    }

    # Task 1: Chart Type
    if check1(js_obj) : 
        chart_type = js_obj['task1']['output']['chart_type']
        chart_variables['chart_type'] = chart_type

    
    if check2(js_obj) : 
        tb = js_obj['task2']['output'].get('text_blocks', [])
        chart_variables['all_txt'] = [block['text'] for block in tb]

    # Axis Titles
    if check3(js_obj) and check4(js_obj) : 
        trole = js_obj['task3']['output'].get('text_roles', [])
        axs_title_id = [block['id'] for block in trole if block['role'] == 'axis_title']
        axs_titles = extract_txt_getRect(tb, axs_title_id)
        
    # Task 4: Axis Information
 
        xax_id = [block['id'] for block in js_obj['task4']['output'].get('axes', {}).get('x-axis', [])]
        xax_ctr = get_combined_centroid(tb, xax_id) if len(xax_id) > 0 else None
    
        yax_id = [block['id'] for block in js_obj['task4']['output'].get('axes', {}).get('y-axis', [])]
        yax_ctr = get_combined_centroid(tb, yax_id) if len(yax_id) > 0 else None

        # Analyze ticks for both axes
        yticks = [convert_to_numeric_or_string(block['text']) for block in tb if block['id'] in yax_id]
        chart_variables['y_ticks'] = yticks if len(yax_id) > 0 else None
        yrange = analyze_array(yticks) if len(yax_id) > 0 else None
    
        xticks = [convert_to_numeric_or_string(block['text']) for block in tb if block['id'] in xax_id]
        chart_variables['x_ticks'] = xticks if len(xax_id) > 0 else None
        xrange = analyze_array(xticks) if len(xax_id) > 0 else None

        # Handle categorical vs numerical axis data
        if isinstance(xrange, tuple):
            chart_variables['xmin'], chart_variables['xmax'] = xrange
            chart_variables['categorical'] = False
        else:
            chart_variables['categorical'] = True
            # Ensure random choice doesn't pick the same text for both min and max
            if chart_variables['all_txt']:
                chart_variables['xmin'] = chart_variables['xmax'] = random.choice(chart_variables['all_txt'])

        if isinstance(yrange, tuple):
            chart_variables['ymin'], chart_variables['ymax'] = yrange
        
        axs_assignments = assign_axis_title(xax_ctr, yax_ctr, axs_titles, js_obj['task4']['output']['_plot_bb'])
        chart_variables.update(axs_assignments)


    # Task 5: Legend Information
    if check5(js_obj) : 
        num_ds = len(js_obj['task5']['output']['legend_pairs'])
        lid = [block['id'] for block in js_obj['task5']['output']['legend_pairs']]
        llbls = [block['text'] for block in tb if block['id'] in lid]
        chart_variables['number_of_ds'] = num_ds
        chart_variables['legend_labels'] = llbls

    return chart_variables

##########################################################

import Levenshtein

def evaluate_text_similarity(prediction_text, reference_text):
    # Calculate edit distance
    # print('\n In evaluate_text_similarity ---',  prediction_text, reference_text)
    edit_distance = Levenshtein.distance(prediction_text, reference_text)

    # Calculate precision, recall, and F1 score for edit distance
    max_length = max(len(prediction_text), len(reference_text))
    precision_edit = 1 - (edit_distance / max_length) if max_length != 0 else 0
    # recall_edit = 1 - (edit_distance / max_length) if max_length != 0 else 0
    # Since precision and recall are equal in this method, the formula simplifies to:
    # f1_score_edit = precision_edit  # Because precision_edit == recall_edit in this calculation
    # return edit_distance, precision_edit, recall_edit, f1_score_edit
    return  precision_edit



def evaluate_chart_type(predicted_answer, actual_answer):
    """Extracts the chart type from the predicted answer."""
    match = re.search(r'the type of chart is (.+?)\.', predicted_answer, re.IGNORECASE)
    if match:
        ans =  match.group(1).strip()
    else : 
        ans = ""
    precision = evaluate_text_similarity(ans, actual_answer)
    return precision


 

def evaluate_daxis_title(predicted, actual_answer):
    # Simple case-insensitive comparison
    # print('\n\n in D axis title', predicted)
    match = re.search(r'The dependant axis is labeled as (.+?)\.', predicted_answer, re.IGNORECASE)
    # print(match)
    if match:
        ans =  match.group(1).strip()
    else : 
        ans = ""
    precision= evaluate_text_similarity(ans, actual_answer)
    return precision




def evaluate_axis_title(predicted, actual_answer):
    # Simple case-insensitive comparison
    # print('\n\n In evaluate_axis_title')
    
    match = re.search(r'the independant axis is labeled as (.+?)\.', predicted_answer, re.IGNORECASE)
    if match:
        ans =  match.group(1).strip()
    else : 
        print(predicted.lower(),  actual_answer.lower())
        ans = ""
    precision = evaluate_text_similarity(ans, actual_answer)

    return precision

def evaluate_yrange(predicted, actual_answer):
    # print('\n\n IN evaluate_yrange')
    pattern = r"the dependant axis ranges from a minimum of (.+?) to a maximum of (.+?) in (.+?)\."
    match = re.search(pattern, predicted, re.IGNORECASE)
    if not match:
        print('! MATCH NOT FOUND')
        print(predicted, actual_answer)
        return 0, 0, 0 
    ymin = match.group(1).strip()
    ymax = match.group(2).strip()
    y_title = match.group(3).strip()
    p1, p2, p3 = 0, 0, 0 
    # print(type(ymin), ymax, y_title, type(actual_answer[0]))
    p1 = evaluate_text_similarity(ymin, str(actual_answer[0]))
    p2 = evaluate_text_similarity(ymax, str(actual_answer[1]))
    p3= evaluate_text_similarity(y_title, actual_answer[2])
    # print(p1, p2, p3)
    return p1, p2, p3 

def evaluate_xrange(predicted, actual_answer):
    # print('\n\n In evaluate_xrange')
    pattern = r"the independant axis ranges from a minimum of (.+?) to a maximum of (.+?) in (.+?)\."
    match = re.search(pattern, predicted, re.IGNORECASE)
    if not match:
        print('!889! MATCH NOT FOUND')
        print(predicted, actual_answer)
        return 0, 0, 0 
    xmin = match.group(1).strip()
    xmax = match.group(2).strip()
    x_title = match.group(3).strip()
    p1, p2, p3 = 0, 0, 0 
    p1 = evaluate_text_similarity(xmin, str(actual_answer[0]))
    p2 = evaluate_text_similarity(xmax, str(actual_answer[1]))
    p3 = evaluate_text_similarity(x_title, actual_answer[2])
    return p1, p2, p3 

import string


def evaluate_categorical_axis(predicted, actual_answer):
    """Evaluates if the predicted categorical labels match the actual labels."""
    # print('\n\nIN evaluate_categorical_axis')
    predicted = ''.join(filter(lambda x: x in string.printable, predicted))[51:]
    # pattern = r"The independant axis is categorical with the labels \[(.+?)\]"
    # match = re.search(pattern, str(predicted), re.IGNORECASE)
    # print(pattern, match)
    # if not match:
    #     print('!! MATCH NOT FOUND')
    #     print(type(predicted), predicted, actual_answer)
    #     return 0, 0
    # predicted_labels_str = match.group(1).split(',')
    # print(predicted)
    predicted_labels_str = predicted.split(',')
    actual_labels = actual_answer[1]
    # Normalize predicted labels (remove quotes and extra spaces)
    predicted_labels_str = [label.strip().strip("'").strip('"') for label in predicted_labels_str]
    # Sort actual labels for consistent comparison
    actual_labels_str = sorted([str(label) for label in actual_labels])

    # For each predicted label, find the best match score among all actual labels
    scores = []
    for pred_label in predicted_labels_str:
        best_score = max([evaluate_text_similarity(pred_label, actual_label) for actual_label in actual_labels_str])
        scores.append(best_score)

    # Calculate the average score
    avg_score = sum(scores) / len(scores) if scores else 0
    return 1, avg_score

def evaluate_ds_count(predicted, actual_count):
    pattern = r'the chart contains a legend that differentiates between the (\d+) data series'
    match = re.search(pattern, predicted, re.IGNORECASE)
    if not match:
        print('!@! MATCH NOT FOUND')
        return 0
    else :
        return int(int(match.group(1).strip()) ==   actual_count)


def evaluate_legend_labels(predicted_answer, actual_labels):
    """Evaluates if the predicted legend labels match the actual labels using text similarity."""
    pattern = r"each data series in the legend corresponds to a unique representation on the chart .+? and has the labels \[(.+?)\]"
    match = re.search(pattern, predicted_answer, re.IGNORECASE)
    if not match:
        return 0  # No match found, return the lowest score

    # Extract predicted labels and normalize them
    predicted_labels_str = match.group(1).split(',')
    predicted_labels_str = [label.strip().strip("'").strip('"') for label in predicted_labels_str]
    # Normalize actual labels for comparison
    actual_labels_str = [str(label) for label in actual_labels]
    # print('\n\n legend')
    # print(predicted_labels_str)
    # print(actual_labels_str)


    # For each predicted label, find the best match score among all actual labels
    scores = []
    for pred_label in predicted_labels_str:
        best_score = max([evaluate_text_similarity(pred_label, actual_label) for actual_label in actual_labels_str])
        scores.append(best_score)

    # print(scores)
    # Calculate the average score
    avg_score = sum(scores) / len(scores) if scores else 0
    return avg_score


evaluation_results = {
    'chart_type' :[], 
    'y_title' :[], 
    'y_min' :[], 
    'y_max' :[],
    'x_min' :[], 
    'x_max' :[], 
    'y_rtitle' :[], 
    'x_rtitle' :[], 
    'x_ticks_label' :[],
    'x_ticks_categorical' : [], 
    'legend_presence_count' :[], 
    'legend_labels_match' :[], 
}
import tqdm 
for pred in tqdm.tqdm(os.listdir(pred_dir)) :
    # pred  = 'PMC3522381___g002.json'
    predictions = json.load(open(os.path.join(pred_dir, pred), 'r'))
    gt = json.load(open(os.path.join(json_dir, pred), 'r'))
    # print(predictions)
    actual_chart_vars = extract_chart_variables(gt)
    # print(actual_chart_vars)
    # print('\n')
    # exit()
    for prediction in predictions:
        # print('\n')
        # print('_'*80)
        question = prediction['question'][0]
        # print('question',question)
        predicted_answer = prediction['predicted_answer']
        # print('predicted_answer', predicted_answer)

        # if 'type of chart' in question:
        if question == 'What is the type of chart ?':
            result = evaluate_chart_type(predicted_answer,  actual_chart_vars['chart_type'].lower())
            # evaluation_results.append((prediction['qa_id'], 'chart_type', result))
            evaluation_results['chart_type'].append(result)
        
        elif question == 'What is the label of the dependant axis in the chart ?':
            result = evaluate_daxis_title(predicted_answer.lower(), actual_chart_vars['y_title'].lower())
            # evaluation_results.append((prediction['qa_id'], 'y_title', result))
            evaluation_results['y_title'].append(result)

        elif question == 'What is the label of the independant axis in the chart ?':
            result = evaluate_axis_title(predicted_answer.lower(), actual_chart_vars['x_title'].lower())
            # evaluation_results.append((prediction['qa_id'], 'x_title', result))
            evaluation_results['y_title'].append(result)
        
        elif question == 'What is the range and title of the dependant axis in the chart ?':
            result = evaluate_yrange(predicted_answer.lower(), (actual_chart_vars['ymin'], actual_chart_vars['ymax'], actual_chart_vars['y_title'].lower()))
            # evaluation_results.append((prediction['qa_id'], 'y_min', result[0]))
            # evaluation_results.append((prediction['qa_id'], 'y_max', result[1]))
            # evaluation_results.append((prediction['qa_id'], 'y_rtitle', result[2]))
            evaluation_results['y_min'].append(result[0])
            evaluation_results['y_max'].append(result[1])
            evaluation_results['y_rtitle'].append(result[2])

        
        
        elif question == 'What is the range and title of the independant axis in the chart ?':
            result = evaluate_xrange(predicted_answer.lower(), (actual_chart_vars['xmin'], actual_chart_vars['xmax'], actual_chart_vars['x_title'].lower()))
            # evaluation_results.append((prediction['qa_id'], 'x_min', result[0]))
            # evaluation_results.append((prediction['qa_id'], 'x_max', result[1]))
            # evaluation_results.append((prediction['qa_id'], 'x_rtitle', result[2]))
            evaluation_results['x_min'].append(result[0])
            evaluation_results['x_max'].append(result[1])
            evaluation_results['x_rtitle'].append(result[2])
        
        elif question == 'Is the independant axis categorical ? What are the tick labels?':
            result = evaluate_categorical_axis(predicted_answer, (actual_chart_vars['categorical'],    actual_chart_vars['x_ticks']))
            # evaluation_results.append((prediction['qa_id'], 'x_ticks_categorical', result[0]))
            # evaluation_results.append((prediction['qa_id'], 'x_ticks_label', result[1]))
            evaluation_results['x_ticks_categorical'].append(result[0])
            evaluation_results['x_ticks_label'].append(result[1])
        
        
        elif question == 'Is there a legend in the chart ? What are the number of dataseries plotted ?':
            result = evaluate_ds_count(predicted_answer, actual_chart_vars.get('number_of_ds', 0))
            # evaluation_results.append((prediction['qa_id'], 'legend_presence_count', result))
            evaluation_results['legend_presence_count'].append(result)
        
        elif question == 'What is the legend label for each data series in the chart, dot they match ?':
            result = evaluate_legend_labels(predicted_answer, actual_chart_vars.get('legend_labels', []))
            # evaluation_results.append((prediction['qa_id'], 'legend_labels_match', result))
            evaluation_results['legend_labels_match'].append(result)

    
    # Print or process the evaluation results as needed'
    # print('\n')        
for res in evaluation_results:
        # print(f"QA ID: {res[0]}, Metric: {res[1]}, Result: {res[2]}")
    print(res, round(sum(evaluation_results[res]), 2), len(evaluation_results[res]), round(sum(evaluation_results[res])/len(evaluation_results[res]),2 ))
    # exit()
