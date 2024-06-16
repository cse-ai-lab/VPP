import torch
import argparse
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
# from transformers import (T5TokenizerFast as T5Tokenizer)
from transformers.optimization import Adafactor, get_cosine_schedule_with_warmup
from dataset.rqa_dataset import ChartConvo as ChartConvoProcessor  
from torch.utils.data import DataLoader, random_split
import os
import tqdm
import wandb
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import uuid
import Levenshtein
import json
import sacrebleu 

parser = argparse.ArgumentParser(description="RQA")
# parser.add_argument("--conv_json", help="Path to the convo jsons", default='/home/csgrad/sahmed9/reps/chartinfo-cqa/dataset/cqa_22_with_id/instructions_test_data_mini.json')
# parser.add_argument("--conv_json", help="Path to the convo jsons", default='/home/csgrad/sahmed9/reps/chartinfo-cqa/dataset/cqa_22_with_id/instructions_qa_test9357.json')
# parser.add_argument("--conv_json", help="Path to the convo jsons", default='/home/csgrad/sahmed9/reps/chartinfo-cqa/dataset/cqa_22_with_id/test_dense_predict.json')
parser.add_argument("--conv_json", help="Path to the convo jsons", default='/home/csgrad/sahmed9/reps/chartinfo-cqa/dataset/cqa_22_with_id/test_binary_premise4.json')



parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--train", action='store_true')
parser.add_argument("--epoch", default=0, type=int, help="Number of epochs")

parser.add_argument("-output", "--output", help="Path to the output file", default='/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv/')
parser.add_argument("-checkpoint_interval", "--checkpoint_interval", default=100, type=int, help="Steps interval to save checkpoints")
parser.add_argument("-ckpt_path", "--ckpt_path", help="Path to a checkpoint file to resume training", default=None)

parser.add_argument("--model_to_use", default='google/matcha-chartqa', help="HF Model to load")
args = parser.parse_args()
ln = '-'*180
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# Function to calculate BLEU score
def calculate_bleu_score(pred,label):
    # Calculate BLEU score
    # bleu_score = sacrebleu.corpus_bleu(pred,label)
    bleu_score = sacrebleu.sentence_bleu(pred, label, smooth_method="exp")
    
    return bleu_score.score


def calculate_f1_score(prediction_tokens, reference_tokens, prediction_text, reference_text ):
    # Calculate precision, recall, and F1 score for token matching
    # print(set(prediction_tokens), set(reference_tokens))
    true_positives = len(set(prediction_tokens) & set(reference_tokens))
    if len(prediction_tokens) == 0 or len(reference_tokens) == 0:
        return 0, 0, 0  # Avoid division by zero
    precision = true_positives / len(prediction_tokens)
    recall = true_positives / len(reference_tokens)
    f1_score_token = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    # exit()

    # Calculate edit distance
    edit_distance = Levenshtein.distance(prediction_text, reference_text)

    # Calculate precision, recall, and F1 score for edit distance
    max_length = max(len(prediction_text), len(reference_text))
    precision_edit = 1 - (edit_distance / max_length) if max_length != 0 else 0
    recall_edit = 1 - (edit_distance / max_length) if max_length != 0 else 0
    f1_score_edit = 2 * (precision_edit * recall_edit) / (precision_edit + recall_edit) if (precision_edit + recall_edit) != 0 else 0

    return f1_score_token, f1_score_edit


def runeval(ckp_):
    # local_rank = int(os.getenv('LOCAL_RANK', '0'))
    # torch.cuda.set_device(local_rank)
    # torch.distributed.init_process_group(backend='gloo')
    local_rank =  torch.device("cuda")
    # print('ARGSPACE')
    # print([k for k in args])
    
    model_name = args.model_to_use
    print('Starting MODEL :: ', model_name)
    model = Pix2StructForConditionalGeneration.from_pretrained(model_name).cuda()
    # model = DistributedDataParallel(model, device_ids=[local_rank])
    processor = Pix2StructProcessor.from_pretrained(args.model_to_use)
    tokenizer = processor.tokenizer #T5Tokenizer.from_pretrained('t5-base', model_max_length=2048)
    # print('model', model)
    print('processor', processor)

    print('Initialize DATASET :: ', args.conv_json)
    dataset = ChartConvoProcessor(args.conv_json)
    # test_sampler = DistributedSampler(dataset, num_replicas=torch.distributed.get_world_size(), rank=local_rank)
    # test_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=test_sampler, collate_fn=ChartConvoProcessor.custom_collate)
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=ChartConvoProcessor.custom_collate)

    output_dir = args.output
    if args.ckpt_path is None : 
       args.ckpt_path = ckp_ 
    if args.ckpt_path is None : 
       rp_title = 'Baseline_Model'
    if args.ckpt_path is not None:
        print(f"Resuming training from checkpoint: {args.ckpt_path}")
        checkpoint = torch.load(args.ckpt_path, map_location=lambda storage, loc: storage.cuda(local_rank))
        new_state_dict = {}
        for key, value in checkpoint['model_state_dict'].items():
            if key.startswith('module.'):
                new_key = key[len('module.'):]  # Remove the 'module.' prefix
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        model.load_state_dict(new_state_dict)
        rp_title = args.ckpt_path.split('/')[-1]
        
    model.eval()
    print('Start Eval')
    
    # Initialize variables to keep track of evaluation results
    total_samples = 0
    total_bleu_score = 0
    total_f1_token_score = 0
    total_f1_edit_score = 0
    worst_score = float('inf')
    worst_case_ids = []

    ln = '-'*180
    report_file =f"\n\n Evaluation Report:Model Checkpoint: {rp_title} {ln} \n\n"
    opdir = '/home/csgrad/sahmed9/reps/RealCQA/code/eval_result/binary_struct_148000_4'
    for data_batch in tqdm.tqdm(test_loader):
        image, question, answer, ids = data_batch

        # print('question', question)
        # print('answer', answer)
        # print('ids', ids)
        
        encoding = processor(images=image, text=question, return_tensors="pt", padding=True, truncation=True, max_length=2048, add_special_tokens=True)
        encoding = {k: v.to(local_rank) for k, v in encoding.items()}
        labels = tokenizer(answer, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids.squeeze().to(local_rank)
        labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask pad tokens in labels
        encoding['labels'] = labels
        # outputs = model.module.generate(**encoding, max_new_tokens=1024)
        outputs = model.generate(**encoding, max_new_tokens=1024)
        # print(outputs)

        # Decode tokens to text for edit distance
        prediction_text = processor.decode(outputs[0], skip_special_tokens=True)
        reference_text = processor.decode(labels, skip_special_tokens=True)
        print('\n\n', prediction_text,"--->>\n" ,reference_text)
        op_fl_name = os.path.join(opdir, ids[0][-1].split('/')[-1][:-4]+'.json')
        # print('op_fl_name', op_fl_name)
        # print('id', ids[0][0])
        # exit()

        ####### ===============>>>> FOR QA OUTPUT 
        # op_fl_name = os.path.join(opdir, ids[0][-1].split('/')[-1][:-4]+'.json')
        # if os.path.exists(op_fl_name) :
        #     temp = json.load(open(op_fl_name, 'r'))
        #     temp.extend([{
        #         "qa_id": ids[0][0],
        #         "predicted_answer": prediction_text}])
        # else :
        #     temp = [{
        #         "qa_id": ids[0][0],
        #         "predicted_answer": prediction_text}] 

        # with open(op_fl_name, 'w') as f :
            # json.dump(temp, f) 




        ####### ===============>>>> FOR Dense OUTPUT /b
        op_fl_name = os.path.join(opdir, ids[0][-1].split('/')[-1][:-4]+'.json')
        if os.path.exists(op_fl_name) :
            temp = json.load(open(op_fl_name, 'r'))
            temp.extend([{
                "qa_id": ids[0][0],
                'question' : question,
                "predicted_answer": prediction_text}])
        else :
            temp = [{
                "qa_id": ids[0][0],
                'question' : question,
                "predicted_answer": prediction_text}] 

        with open(op_fl_name, 'w') as f :
            json.dump(temp, f) 

        # Calculate evaluation metrics
        bleu_score = calculate_bleu_score( prediction_text, [reference_text] )

        f1_token_score, f1_edit_score = calculate_f1_score(set(sorted([k.cpu().item() for k in outputs[0]])), set(sorted([k.cpu().item() for k in labels])), prediction_text, reference_text )

        # Update total scores
        total_samples += len(data_batch)
        total_bleu_score += bleu_score
        total_f1_token_score += f1_token_score
        total_f1_edit_score += f1_edit_score

        # Track worst case
        if bleu_score < worst_score:
            worst_score = bleu_score
            worst_case_ids = ids

        # report_file+=f"\nSample ID: {ids}, BLEU Score: {bleu_score}, F1 Score (Token Matching): {f1_token_score}, F1 Score (Edit Distance): {f1_edit_score}\n"

    
    # Check if the report file already exists
    report_file_exists = os.path.exists("evaluation_report.txt")

    # Open the report file in append mode if it already exists
    report_file_mode = "a" if report_file_exists else "w"

    
    


    # Calculate average scores
    avg_bleu_score = total_bleu_score / total_samples
    avg_f1_token_score = total_f1_token_score / total_samples
    avg_f1_edit_score = total_f1_edit_score / total_samples

    # Write evaluation report to text file
        
    report_file+=f"{ln}\n\nTotal Samples: {total_samples}\n Average BLEU Score: {avg_bleu_score}\nAverage F1 Score (Token Matching): {avg_f1_token_score}\nAverage F1 Score (Edit Distance): {avg_f1_edit_score}\nWorst Case ID(s): {worst_case_ids}\n {ln} "
    # Open text file to write the evaluation report
    ln = "-"*180
    with open("evaluation_report.txt", report_file_mode) as f :
        f.write(report_file)

        
            

    # Print out the evaluation report
    print("Evaluation Report:")
    print(f"Total Samples: {total_samples}")
    print(f"Average BLEU Score: {avg_bleu_score}")
    print(f"Average F1 Score (Token Matching): {avg_f1_token_score}")
    print(f"Average F1 Score (Edit Distance): {avg_f1_edit_score}")
    print(f"Worst Case ID(s): {worst_case_ids}")
                
        


if __name__ == "__main__":
    # run()
    # ckpt = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv/checkpoint_epoch_1_step_1000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv2/checkpoint_epoch_1_step_1000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv2/checkpoint_epoch_1_step_2000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv2/checkpoint_epoch_1_step_3000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv2/checkpoint_epoch_1_step_4000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv2/checkpoint_epoch_1_step_5000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv2/checkpoint_epoch_1_step_6000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv2/checkpoint_epoch_1_step_7000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv2/checkpoint_epoch_1_step_8000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv2/checkpoint_epoch_1_step_9000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv2/checkpoint_epoch_1_step_10000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv2/checkpoint_epoch_1_step_11000.pt'    
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv2/checkpoint_epoch_1_step_12000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv2/checkpoint_epoch_1_step_13000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv2/checkpoint_epoch_1_step_14000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv2/checkpoint_epoch_1_step_15000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv2/checkpoint_epoch_1_step_16000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv2/checkpoint_epoch_1_step_17000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv2/checkpoint_epoch_1_step_18000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv2/checkpoint_epoch_1_step_19000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv2/checkpoint_epoch_1_step_20000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv2/checkpoint_epoch_1_step_21000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv2/checkpoint_epoch_1_step_22000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv2/checkpoint_epoch_1_step_23000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv2/checkpoint_epoch_1_step_24000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv2/checkpoint_epoch_1_step_25000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv2/checkpoint_epoch_1_step_26000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv2/checkpoint_epoch_1_step_27000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv2/checkpoint_epoch_1_step_28000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv2/checkpoint_epoch_1_step_29000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv3/checkpoint_epoch_2_step_30000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv3/checkpoint_epoch_2_step_31000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv3/checkpoint_epoch_2_step_32000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv3/checkpoint_epoch_2_step_33000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv3/checkpoint_epoch_2_step_34000.pt'
    
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv3/checkpoint_epoch_2_step_35000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv3/checkpoint_epoch_2_step_36000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv3/checkpoint_epoch_2_step_37000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv3/checkpoint_epoch_2_step_38000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv3/checkpoint_epoch_2_step_39000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv3/checkpoint_epoch_2_step_40000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv4/checkpoint_epoch_3_step_41000.pt'
    
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv4/checkpoint_epoch_3_step_43000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv4/checkpoint_epoch_3_step_44000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv4/checkpoint_epoch_3_step_45000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv4/checkpoint_epoch_3_step_46000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv4/checkpoint_epoch_3_step_47000.pt'
    
    
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv4/checkpoint_epoch_3_step_48000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv4/checkpoint_epoch_3_step_49000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv4/checkpoint_epoch_3_step_50000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv4/checkpoint_epoch_3_step_51000.pt'
    
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv4/checkpoint_epoch_3_step_52000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv4/checkpoint_epoch_3_step_53000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv4/checkpoint_epoch_3_step_54000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv4/checkpoint_epoch_3_step_55000.pt'

    
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv4/checkpoint_epoch_3_step_42000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv4/checkpoint_epoch_3_step_56000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv4/checkpoint_epoch_3_step_57000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv4/checkpoint_epoch_3_step_58000.pt'
    
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv4/checkpoint_epoch_3_step_59000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv4/checkpoint_epoch_3_step_60000.pt'
    
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv4/checkpoint_epoch_3_step_61000.pt'
    
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv5/checkpoint_epoch_4_step_62000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv5/checkpoint_epoch_4_step_63000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv5/checkpoint_epoch_4_step_64000.pt'
    
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv5/checkpoint_epoch_4_step_66000.pt'
    
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv5/checkpoint_epoch_4_step_67000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv5/checkpoint_epoch_4_step_68000.pt'
    
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv5/checkpoint_epoch_4_step_69000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv5/checkpoint_epoch_4_step_70000.pt'
    
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv5/checkpoint_epoch_4_step_72000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv5/checkpoint_epoch_4_step_73000.pt'

    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv5/checkpoint_epoch_4_step_74000.pt'
    
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv5/checkpoint_epoch_4_step_65000.pt'    
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv5/checkpoint_epoch_4_step_76000.pt'
        
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv5/checkpoint_epoch_4_step_77000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv5/checkpoint_epoch_4_step_78000.pt'
    
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv5/checkpoint_epoch_4_step_79000.pt'
    
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv5/checkpoint_epoch_4_step_80000.pt'
    
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv5/checkpoint_epoch_4_step_92000.pt'

    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv5/checkpoint_epoch_4_step_93000.pt'
    
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv5/checkpoint_epoch_4_step_94000.pt'
    
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv5/checkpoint_epoch_4_step_95000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv5/checkpoint_epoch_4_step_96000.pt'
    
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv5/checkpoint_epoch_4_step_97000.pt'
    
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv5/checkpoint_epoch_4_step_98000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv5/checkpoint_epoch_4_step_99000.pt'
    
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv5/checkpoint_epoch_4_step_100000.pt'
    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv5/checkpoint_epoch_4_step_101000.pt'


    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv_done/checkpoint_epoch_7_step_142000.pt'

    ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv_done/checkpoint_epoch_7_step_148000.pt'

    # ckp_ = '/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv_done/checkpoint_epoch_7_step_150000.pt'
    
    runeval(ckp_)
    