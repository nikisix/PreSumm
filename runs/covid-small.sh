# Step 1;
java edu.stanford.nlp.pipeline.StanfordCoreNLP \
    -annotators tokenize,ssplit \
    -ssplit.newlineIsSentenceBreak always \
    -filelist mapping_for_corenlp.txt \
    -outputFormat json \
    -outputDirectory ./results/covid-2-scratch/

    
# STEP 3
# Output files will be in the json directory
python src/preprocess.py \
    --mode custom_format_to_lines \
    --raw_path results/covid-2-scratch/ \
    --save_path json_data_covid \
    --n_cpus 1 \
    --use_bert_basic_tokenizer false \
    --map_path urls \
    --log_file logs/covid.log


# STEP 4
# output to bert_data_covid/train.0.bert.pt
# if this file exists it will be a no-op, so delete the file before running
python src/preprocess.py \
    --mode custom_format_to_bert \
    --raw_path ./json_data_covid \
    --save_path ./bert_data_covid \
    --lower \
    --n_cpus 1 \
    --log_file ./logs/covid.log

# Step 
python src/train.py \
    --task abs \
    --mode test_text \
    --ext_dropout 0.1 \
    --lr .002\
    --report_every 50 \
    --save_checkpoint_steps 99 \
    --batch_size 3000 \
    --accum_count 2 \
    --log_file logs/ext_bert_cnndm \
    --use_interval true \
    --max_pos 30000 \
    --warmup_steps 5 \
    --train_steps 20 \
    --visible_gpus 0 \
    --temp_dir ./temp \
    --model_path models/ \
    --result_path results \
    --bert_data_path bert_data/abs_cnndm_sample \
    --text_src raw_data_covid/full_text_small_head.txt \
    --test_from models/model_step_148000.pt

# when upping the max_pos to 30000
# ERROR: size mismatch for bert.model.embeddings.position_embeddings.weight: copying a param with shape torch.Size([512, 768]) from checkpoint, the shape in current model is torch.Size([30000, 768]).

python src/train.py \
    --task abs \
    --mode test_text \
    --ext_dropout 0.1 \
    --lr .002\
    --report_every 50 \
    --save_checkpoint_steps 99 \
    --batch_size 3000 \
    --accum_count 2 \
    --log_file logs/ext_bert_cnndm \
    --use_interval true \
    --max_pos 30000 \
    --warmup_steps 5 \
    --train_steps 20 \
    --visible_gpus 0 \
    --temp_dir ./temp \
    --model_path models/ \
    --result_path results \
    --bert_data_path bert_data/abs_cnndm_sample \
    --text_src raw_data/temp.raw_src \
    --text_tgt raw_data/temp.raw_tgt \
    --test_from models/model_step_148000.pt

# Custom Fine-tuning
python src/train.py \
    --task abs \
    --mode train \
    --ext_dropout 0.1 \
    --lr .002\
    --report_every 50 \
    --save_checkpoint_steps 99 \
    --batch_size 3000 \
    --accum_count 2 \
    --log_file logs/ext_bert_cnndm \
    --use_interval true \
    --max_pos 30000 \
    --warmup_steps 5 \
    --train_steps 20 \
    --visible_gpus 0 \
    --temp_dir ./temp \
    --model_path models/ \
    --result_path results \
    --bert_data_path bert_data_covid/train.0.bert.pt \

# Currently stuck on the following error
# RuntimeError: CUDA out of memory. Tried to allocate 2.00 MiB (GPU 0; 3.95 GiB total capacity; 3.15 GiB already allocated; 3.00 MiB free; 155.55 MiB cached)
