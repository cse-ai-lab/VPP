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
import gc

def generate_unique_id():
    return str(uuid.uuid4())
# Setup
parser = argparse.ArgumentParser(description="RQA")
# parser.add_argument("--conv_json", help="Path to the convo jsons", default='/home/csgrad/sahmed9/reps/chartinfo-cqa/dataset/cqa_22_with_id/instructions_train_data_clean.json')
parser.add_argument("--conv_json", help="Path to the convo jsons", default='/home/csgrad/sahmed9/reps/chartinfo-cqa/dataset/cqa_22_with_id/DP_RP/DP_RP.json')

parser.add_argument("--batch_size", default=20, type=int)
parser.add_argument("--train", action='store_true')
parser.add_argument("--epoch", default=3, type=int, help="Number of epochs")

parser.add_argument("-output", "--output", help="Path to the output file", default='/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv/')
parser.add_argument("-checkpoint_interval", "--checkpoint_interval", default=100, type=int, help="Steps interval to save checkpoints")
parser.add_argument("-ckpt_path", "--ckpt_path", help="Path to a checkpoint file to resume training", default=None)
# parser.add_argument("-ckpt_path", "--ckpt_path", help="Path to a checkpoint file to resume training", default='reps/RealCQA/code/outputChartConv_done/checkpoint_epoch_7_step_148000.pt')


parser.add_argument("--model_to_use", default='google/matcha-chartqa', help="HF Model to load")
args = parser.parse_args()

os.environ["WANDB_PROJECT"] = "chart-P2S"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3,2"
wandb.init(project="chart-P2S", config=args)



def compute_grad_norm(model, norm_type=2):
    """
    Computes the total norm of the gradients for all trainable parameters of the model.
    Args:
        model (torch.nn.Module): The model for which gradients are computed.
        norm_type (float or int): Type of the used p-norm. Can be `'fro'`, `'nuc'`, or any positive real number yielding the corresponding p-norm. Default is 2.

    Returns:
        total_norm (float): Total norm of the model's gradients.
    """
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

def save_checkpoint(model, optimizer, scheduler, epoch, step, output_dir):
    if scheduler :
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'step': step
        }
    else : 
        checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'step': step
    }
    checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch}_step_{step}.pt")
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")



def run():
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='gloo')

    # print('ARGSPACE')
    # print([k for k in args])
    
    model_name = args.model_to_use
    print('Starting MODEL :: ', model_name)
    model = Pix2StructForConditionalGeneration.from_pretrained(model_name).cuda()
    model = DistributedDataParallel(model, device_ids=[local_rank])
    processor = Pix2StructProcessor.from_pretrained(args.model_to_use)
    tokenizer = processor.tokenizer #T5Tokenizer.from_pretrained('t5-base', model_max_length=2048)
    # print('model', model)
    print('processor', processor)

    print('Initialize DATASET :: ')
    dataset = ChartConvoProcessor(args.conv_json)
    train_size = int(0.999 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_sampler = DistributedSampler(train_dataset, num_replicas=torch.distributed.get_world_size(), rank=local_rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=torch.distributed.get_world_size(), rank=local_rank, shuffle=True)


    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, collate_fn=ChartConvoProcessor.custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, collate_fn=ChartConvoProcessor.custom_collate)


    print('Starting OPTIMIZER :: ')
    # optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    # optimizer = Adafactor(model.parameters(), scale_parameter=False, relative_step=False, lr=1e-5, weight_decay=1e-05)
    optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, weight_decay=1e-05)
    # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=400000)
    scheduler = None
    wandb.config.update({
        "relative_step": True,
        "weight_decay": 1e-05,
        "scale_parameter": True,
        "initial_lr": 1e-5,  
        "num_steps": 400000,  # Total training steps if you wish to log it for reference
        "warmup_steps": 1000,  # Only applicable if you had a manual warmup phase, might not apply here
    })

    print(optimizer)

    # output_dir = os.path.join(args.output, model_name, generate_unique_id())
    # os.makedirs(output_dir, exist_ok=True)
    print('Output Directory is : ', args.output)
    start_epoch = 0
    start_step = 0
   
    if args.ckpt_path is not None:
        print(f"Resuming training from checkpoint: {args.ckpt_path}")
        checkpoint = torch.load(args.ckpt_path, map_location=lambda storage, loc: storage.cuda(local_rank))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        start_step = 0 #checkpoint.get('step', 0)  # Use .get in case 'step' isn't in older checkpoints

    print('Start Epoch, Step :', start_epoch, start_step )
    model.train()
    try:
        for epoch in range(start_epoch, args.epoch):
            print('Start Train: Epoch ', epoch)
            train_sampler.set_epoch(epoch)
            for step, data_batch in enumerate(tqdm.tqdm(train_loader, initial=start_step)):
                optimizer.zero_grad()
                image, question, answer, ids = data_batch
                encoding = processor(images=image, text=question, return_tensors="pt", padding=True, truncation=True, max_length=2048, add_special_tokens=True)
                encoding = {k: v.to(local_rank) for k, v in encoding.items()}
                labels = tokenizer(answer, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids.squeeze().to(local_rank)
                labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask pad tokens in labels
                encoding['labels'] = labels
                outputs = model(**encoding)
                loss = outputs.loss
                loss.backward()
                
                grad_norm = compute_grad_norm(model)
                wandb.log({"grad_norm": grad_norm})
                
                optimizer.step()
                
                if scheduler is not None :
                    current_lr = scheduler.get_last_lr()[0]  # get_last_lr() returns a list of learning rates for all parameter groups
                    wandb.log({"learning_rate": current_lr, "step": step})            
                    scheduler.step()
                # # Assuming `optimizer` is your Adafactor instance
                # for i, param_group in enumerate(optimizer.param_groups):
                #     # Example of logging an internal state, adjust based on actual optimizer internals
                #     internal_state = param_group['state'].get('step_size', None)
                #     if internal_state is not None:
                #         wandb.log({f"param_group_{i}_step_size": internal_state})
 
                print('Train loss', loss.item()) 
                wandb.log({"train_loss": loss.item()})
                # Checkpointing every k steps
                # print( (step - 1) % args.checkpoint_interval == 0 and local_rank == 0 ,step + 1, args.checkpoint_interval, (step + 1) % args.checkpoint_interval,  local_rank)
                if (step - 1) % args.checkpoint_interval == 0 and local_rank == 0:
                    save_checkpoint(model, optimizer, scheduler, epoch + 1, start_step + step + 1, args.output)
                    torch.cuda.empty_cache()
                    gc.collect()
                    # exit()
                    
                # Reset start_step to 0 after resuming
                start_step = 0

            # Validation loop
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for data_batch in tqdm.tqdm(val_loader):
                    charts, questions, answers, ids_ = data_batch
                    inputs = processor(images=charts, text=questions, return_tensors="pt", padding=True, truncation=True)
                    inputs = {k: v.to(local_rank) for k, v in inputs.items()} 
                    processed_labels = processor(images=charts, text=answers, return_tensors="pt", padding=True, truncation=True)

                    labels = processed_labels['decoder_input_ids'].to(local_rank)
                    labels[labels == processor.tokenizer.pad_token_id] = -100  # Adjusting for padding token IDs
                    inputs['labels'] = labels

                    optimizer.zero_grad()
                    outputs = model(**inputs)
                    val_loss += outputs.loss.item()

            avg_val_loss = val_loss / len(val_loader)
            print('avg_val_loss', avg_val_loss)
            wandb.log({"val_loss": avg_val_loss})
            model.train()

    except KeyboardInterrupt:
        # Save a checkpoint if training is interrupted
        print("Training interrupted. Saving checkpoint...")
        save_checkpoint(model, optimizer, scheduler, epoch+1, step+1, output_dir)
        print("Checkpoint saved. Exiting training.")

if __name__ == "__main__":
    run()
